#!/usr/bin/env python3

"""
Patient-level synthetic CT signed-error prediction and correction.

Inputs:
    - MRI
    - Synthetic CT (one or multiple per patient)
    - Reference CT
    - Binary mask

Target:
    Signed error = CT - sCT

Correction:
    Corrected sCT = sCT + predicted error

IMPORTANT:
    Input images must already be spatially aligned.
    Users must verify that array[z, :, :] corresponds to axial slices.

Training:
    - 2D FlexibleUNet
    - EfficientNet backbone
    - Masked MAE + L1 regularization
    - AdamW optimizer
    - ReduceLROnPlateau scheduler
    - Early stopping

Outputs:
    - Best model checkpoint
    - Training log (.txt)
    - TensorBoard logs
    - Predicted error maps
    - Corrected sCT volumes
    - Test metrics (.csv)
"""

import argparse
import configparser
import csv
import random
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from monai.networks.nets import FlexibleUNet
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# CONFIG UTILITIES
# ============================================================

def parse_bool(cfg, section, key, fallback=False):

    return cfg.getboolean(
        section,
        key,
        fallback=fallback
    )


def read_ids(path):

    path = Path(path)

    ids = [
        x.strip()
        for x in path.read_text().splitlines()
        if x.strip() and not x.lstrip().startswith("#")
    ]

    if not ids:
        raise ValueError(
            f"No patient IDs found in {path}"
        )

    if len(ids) != len(set(ids)):
        raise ValueError(
            f"Duplicate patient IDs found in {path}"
        )

    return ids


# ============================================================
# IMAGE UTILITIES
# ============================================================

def read_image(path):

    img = sitk.ReadImage(str(path))

    if img.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError(
            f"Only scalar images are supported: {path}"
        )

    return img


def same_geometry(a, b):

    return (
        a.GetSize() == b.GetSize()
        and np.allclose(
            a.GetSpacing(),
            b.GetSpacing(),
            atol=1e-5
        )
        and np.allclose(
            a.GetOrigin(),
            b.GetOrigin(),
            atol=1e-4
        )
        and np.allclose(
            a.GetDirection(),
            b.GetDirection(),
            atol=1e-5
        )
    )


# ============================================================
# PADDING / CROPPING
# ============================================================

def pad_or_crop(array, hw, fill):

    """
    Centre pad/crop in the in-plane dimensions (Y, X).

    Returns:
        processed array
        reversible transform
    """

    h, w = array.shape[-2:]

    th, tw = hw

    sy = max(0, (h - th) // 2)
    sx = max(0, (w - tw) // 2)

    dy = max(0, (th - h) // 2)
    dx = max(0, (tw - w) // 2)

    ny = min(h, th)
    nx = min(w, tw)

    out = np.full(
        (*array.shape[:-2], th, tw),
        fill,
        dtype=np.float32
    )

    out[
        ...,
        dy:dy + ny,
        dx:dx + nx
    ] = array[
        ...,
        sy:sy + ny,
        sx:sx + nx
    ]

    transform = (
        h, w,
        sy, sx,
        dy, dx,
        ny, nx
    )

    return out, transform


def restore(array, transform, fill=0.0):

    """
    Restore the original in-plane dimensions.
    """

    h, w, sy, sx, dy, dx, ny, nx = transform

    out = np.full(
        (*array.shape[:-2], h, w),
        fill,
        dtype=np.float32
    )

    out[
        ...,
        sy:sy + ny,
        sx:sx + nx
    ] = array[
        ...,
        dy:dy + ny,
        dx:dx + nx
    ]

    return out


# ============================================================
# PATIENT LOADING
# ============================================================

def load_patient(pid, cfg, with_ct=True):

    base = Path(
        cfg["paths"]["data_dir"]
    ) / pid

    # --------------------------------------------------------
    # MRI / CT / MASK
    # --------------------------------------------------------

    mri = read_image(
        base / cfg["images"]["mri_filename"]
    )

    mask = read_image(
        base / cfg["images"]["mask_filename"]
    )

    ct = (
        read_image(
            base / cfg["images"]["ct_filename"]
        )
        if with_ct
        else None
    )

    # --------------------------------------------------------
    # GEOMETRY CHECK
    # --------------------------------------------------------

    for name, img in (
        ("mask", mask),
        ("CT", ct)
    ):

        if img is not None and not same_geometry(mri, img):

            raise ValueError(
                f"{pid}: MRI and {name} have different geometry. "
                "Align images externally before running."
            )

    # --------------------------------------------------------
    # SYNTHETIC CT FILES
    # --------------------------------------------------------

    pattern = cfg["images"].get(
        "sct_glob",
        "sCT*.nii.gz"
    )

    sct_dir = Path(
        cfg["paths"].get(
            "sct_dir",
            cfg["paths"]["data_dir"]
        )
    ) / pid

    files = sorted(
        sct_dir.glob(pattern)
    )

    if not files:

        raise FileNotFoundError(
            f"{pid}: no synthetic CT files match "
            f"{sct_dir / pattern}"
        )

    scts = {}

    for f in files:

        sct_img = read_image(f)

        if not same_geometry(mri, sct_img):

            raise ValueError(
                f"{pid}: sCT geometry differs: {f}"
            )

        scts[f.name] = sitk.GetArrayFromImage(
            sct_img
        ).astype(np.float32)

    # --------------------------------------------------------
    # NUMPY ARRAYS
    # --------------------------------------------------------

    mri_arr = sitk.GetArrayFromImage(
        mri
    ).astype(np.float32)

    mn = float(mri_arr.min())
    mx = float(mri_arr.max())

    if mx > mn:

        mri_arr = (
            mri_arr - mn
        ) / (
            mx - mn
        )

    else:

        mri_arr = np.zeros_like(
            mri_arr
        )

    mask_arr = (
        sitk.GetArrayFromImage(mask) > 0
    ).astype(np.float32)

    ct_arr = (
        sitk.GetArrayFromImage(ct).astype(np.float32)
        if ct is not None
        else None
    )

    # --------------------------------------------------------
    # OPTIONAL IDEAL sCT
    # --------------------------------------------------------

    if (
        ct_arr is not None
        and parse_bool(
            cfg,
            "images",
            "include_ideal_sct",
            False
        )
    ):

        ideal_name = "__ideal_CT__.nii.gz"

        if ideal_name in scts:

            raise ValueError(
                "Reserved ideal sCT filename collision"
            )

        scts[ideal_name] = ct_arr.copy()

    return (
        mri,
        mri_arr,
        ct_arr,
        mask_arr,
        scts
    )


# ============================================================
# DATASET
# ============================================================

class SliceDataset(Dataset):

    def __init__(
        self,
        patients,
        cfg,
        augment=False
    ):

        self.samples = []
        self.augment = augment

        hw = (
            cfg.getint("model", "height"),
            cfg.getint("model", "width")
        )

        print(
            f"\n[DATASET] Loading {len(patients)} patients...",
            flush=True
        )

        for i, pid in enumerate(patients, start=1):

            print(
                f"[DATASET] Patient {i}/{len(patients)}: {pid}",
                flush=True
            )

            _, mri, ct, mask, scts = load_patient(
                pid,
                cfg,
                with_ct=True
            )

            print(
                f"[DATASET] {pid}: {len(scts)} sCT files",
                flush=True
            )

            for name, sct in scts.items():

                for z in range(mask.shape[0]):

                    if not mask[z].any():
                        continue

                    a, _ = pad_or_crop(
                        mri[z],
                        hw,
                        0
                    )

                    b, _ = pad_or_crop(
                        sct[z],
                        hw,
                        -1000
                    )

                    target, _ = pad_or_crop(
                        ct[z] - sct[z],
                        hw,
                        0
                    )

                    mk, _ = pad_or_crop(
                        mask[z],
                        hw,
                        0
                    )

                    self.samples.append(
                        (
                            np.stack([a, b]),
                            target[None],
                            mk[None]
                        )
                    )

        if not self.samples:

            raise ValueError(
                "No non-empty masked slices found."
            )

        print(
            f"[DATASET] Total samples: {len(self.samples)}",
            flush=True
        )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        inp, target, mask = (
            torch.from_numpy(x.copy()).float()
            for x in self.samples[idx]
        )

        # Random left-right flip

        if self.augment and random.random() < 0.5:

            inp = inp.flip(-1)
            target = target.flip(-1)
            mask = mask.flip(-1)

        return inp, target, mask


# ============================================================
# MODEL
# ============================================================

def make_model(cfg, device):

    model = FlexibleUNet(
        in_channels=2,
        out_channels=1,
        spatial_dims=2,

        backbone=cfg.get(
            "model",
            "backbone",
            fallback="efficientnet-b4"
        ),

        pretrained=parse_bool(
            cfg,
            "model",
            "pretrained",
            True
        ),

        dropout=cfg.getfloat(
            "model",
            "dropout",
            fallback=0.3
        )
    )

    return model.to(device)


# ============================================================
# LOSS
# ============================================================

def loss_fn(
    pred,
    target,
    mask,
    model,
    l1_lambda
):

    masked_mae = (
        (pred - target).abs() * mask
    ).sum() / mask.sum().clamp_min(1)

    l1 = sum(
        p.abs().sum()
        for p in model.parameters()
    )

    return masked_mae + l1_lambda * l1


# ============================================================
# TRAINING
# ============================================================

def train(
    cfg,
    device,
    train_ids,
    val_ids,
    out_dir
):

    seed_everything(
        cfg.getint(
            "training",
            "seed",
            fallback=42
        )
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = make_model(
        cfg,
        device
    )

    lr = cfg.getfloat(
        "training",
        "learning_rate"
    )

    wd = cfg.getfloat(
        "training",
        "weight_decay"
    )

    l1 = cfg.getfloat(
        "training",
        "l1_lambda"
    )

    batch_size = cfg.getint(
        "training",
        "batch_size"
    )

    workers = cfg.getint(
        "training",
        "num_workers",
        fallback=0
    )

    max_epochs = cfg.getint(
        "training",
        "max_epochs"
    )

    patience = cfg.getint(
        "early_stopping",
        "patience"
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=wd
    )

    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.getfloat(
            "scheduler",
            "factor"
        ),
        patience=cfg.getint(
            "scheduler",
            "patience"
        )
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    print(
        "\n[TRAIN] Loading training dataset...",
        flush=True
    )

    train_dataset = SliceDataset(
        train_ids,
        cfg,
        augment=True
    )

    print(
        "\n[VALIDATION] Loading validation dataset...",
        flush=True
    )

    val_dataset = SliceDataset(
        val_ids,
        cfg,
        augment=False
    )

    tr = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers
    )

    va = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers
    )

    # --------------------------------------------------------
    # OUTPUT DIRECTORIES
    # --------------------------------------------------------

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    log_dir = out_dir.parent / "logs"

    log_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # TENSORBOARD
    # --------------------------------------------------------

    writer = SummaryWriter(
        log_dir=str(
            log_dir / "tensorboard"
        )
    )

    # --------------------------------------------------------
    # TXT LOG
    # --------------------------------------------------------

    log_path = log_dir / "training_log.txt"

    log_file = open(
        log_path,
        "w",
        encoding="utf-8"
    )

    log_file.write(
        "SCT SIGNED-ERROR PREDICTION - TRAINING LOG\n"
    )

    log_file.write(
        "=" * 110 + "\n"
    )

    log_file.write(
        f"Device: {device}\n"
    )

    log_file.write(
        f"Training patients: {len(train_ids)}\n"
    )

    log_file.write(
        f"Validation patients: {len(val_ids)}\n"
    )

    log_file.write(
        f"Training samples: {len(train_dataset)}\n"
    )

    log_file.write(
        f"Validation samples: {len(val_dataset)}\n"
    )

    log_file.write(
        f"Batch size: {batch_size}\n"
    )

    log_file.write(
        f"Initial learning rate: {lr}\n"
    )

    log_file.write(
        f"Weight decay: {wd}\n"
    )

    log_file.write(
        f"L1 lambda: {l1}\n"
    )

    log_file.write(
        f"Maximum epochs: {max_epochs}\n"
    )

    log_file.write(
        f"Early stopping patience: {patience}\n"
    )

    log_file.write(
        "=" * 110 + "\n\n"
    )

    log_file.flush()

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    best = float("inf")
    best_epoch = 0
    bad = 0

    print(
        "\n[TRAINING STARTED]\n",
        flush=True
    )

    try:

        for epoch in range(
            1,
            max_epochs + 1
        ):

            # =================================================
            # TRAINING PHASE
            # =================================================

            model.train()

            train_total = 0.0
            train_n = 0

            for inp, tgt, msk in tr:

                inp = inp.to(device)
                tgt = tgt.to(device)
                msk = msk.to(device)

                optimizer.zero_grad()

                pred = model(inp)

                loss = loss_fn(
                    pred,
                    tgt,
                    msk,
                    model,
                    l1
                )

                loss.backward()

                optimizer.step()

                train_total += (
                    loss.item() * len(inp)
                )

                train_n += len(inp)

            train_loss = (
                train_total / train_n
            )

            # =================================================
            # VALIDATION PHASE
            # =================================================

            model.eval()

            val_total = 0.0
            val_n = 0

            with torch.no_grad():

                for inp, tgt, msk in va:

                    inp = inp.to(device)
                    tgt = tgt.to(device)
                    msk = msk.to(device)

                    pred = model(inp)

                    loss = loss_fn(
                        pred,
                        tgt,
                        msk,
                        model,
                        l1
                    )

                    val_total += (
                        loss.item() * len(inp)
                    )

                    val_n += len(inp)

            val_loss = (
                val_total / val_n
            )

            # =================================================
            # SCHEDULER
            # =================================================

            scheduler.step(val_loss)

            current_lr = (
                optimizer.param_groups[0]["lr"]
            )

            # =================================================
            # BEST CHECKPOINT
            # =================================================

            improved = val_loss < best

            if improved:

                best = val_loss
                best_epoch = epoch
                bad = 0

                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "val_loss": best
                    },
                    out_dir / "best.pth"
                )

            else:

                bad += 1

            # =================================================
            # LOG LINE
            # =================================================

            log_line = (
                f"Epoch {epoch:03d}/{max_epochs} | "
                f"Train Loss: {train_loss:.5f} | "
                f"Val Loss: {val_loss:.5f} | "
                f"LR: {current_lr:.2e} | "
                f"Best Val: {best:.5f} | "
                f"Patience: {bad}/{patience}"
            )

            if improved:

                log_line += (
                    " | BEST MODEL SAVED"
                )

            # -------------------------------------------------
            # CONSOLE
            # -------------------------------------------------

            print(
                log_line,
                flush=True
            )

            # -------------------------------------------------
            # TXT
            # -------------------------------------------------

            log_file.write(
                log_line + "\n"
            )

            log_file.flush()

            # -------------------------------------------------
            # TENSORBOARD
            # -------------------------------------------------

            writer.add_scalar(
                "Loss/Train",
                train_loss,
                epoch
            )

            writer.add_scalar(
                "Loss/Validation",
                val_loss,
                epoch
            )

            writer.add_scalar(
                "LearningRate",
                current_lr,
                epoch
            )

            writer.flush()

            # =================================================
            # EARLY STOPPING
            # =================================================

            if bad >= patience:

                message = (
                    f"\n[EARLY STOPPING] "
                    f"No validation improvement "
                    f"for {patience} epochs."
                )

                print(
                    message,
                    flush=True
                )

                log_file.write(
                    message + "\n"
                )

                log_file.flush()

                break

        # =====================================================
        # TRAINING SUMMARY
        # =====================================================

        summary = (
            "\n"
            + "=" * 110
            + "\n"
            + "[TRAINING COMPLETED]\n"
            + f"Best epoch: {best_epoch}\n"
            + f"Best validation loss: {best:.5f}\n"
            + f"Checkpoint: {out_dir / 'best.pth'}\n"
            + "=" * 110
            + "\n"
        )

        print(
            summary,
            flush=True
        )

        log_file.write(
            summary
        )

        log_file.flush()

    finally:

        log_file.close()
        writer.close()

    return out_dir / "best.pth"


# ============================================================
# INFERENCE / TESTING
# ============================================================

@torch.no_grad()
def predict(
    cfg,
    device,
    checkpoint,
    patients,
    out_dir
):

    print(
        "\n[TEST] Loading best checkpoint...",
        flush=True
    )

    model = make_model(
        cfg,
        device
    )

    try:

        ckpt = torch.load(
            checkpoint,
            map_location=device,
            weights_only=True
        )

    except TypeError:

        ckpt = torch.load(
            checkpoint,
            map_location=device
        )

    model.load_state_dict(
        ckpt["state_dict"]
    )

    model.eval()

    print(
        f"[TEST] Checkpoint epoch: {ckpt['epoch']}",
        flush=True
    )

    hw = (
        cfg.getint("model", "height"),
        cfg.getint("model", "width")
    )

    bs = cfg.getint(
        "inference",
        "batch_size",
        fallback=16
    )

    rows = []

    # --------------------------------------------------------
    # PATIENT LOOP
    # --------------------------------------------------------

    for i, pid in enumerate(
        patients,
        start=1
    ):

        print(
            f"\n[TEST] Patient {i}/{len(patients)}: {pid}",
            flush=True
        )

        reference, mri, ct, mask, scts = load_patient(
            pid,
            cfg,
            with_ct=True
        )

        dest = out_dir / pid

        dest.mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------------------------------
        # sCT LOOP
        # ----------------------------------------------------

        for name, sct in scts.items():

            print(
                f"[TEST] Processing sCT: {name}",
                flush=True
            )

            inp_slices = []
            transforms = []

            # ------------------------------------------------
            # PREPARE SLICES
            # ------------------------------------------------

            for z in range(
                sct.shape[0]
            ):

                a, t = pad_or_crop(
                    mri[z],
                    hw,
                    0
                )

                b, _ = pad_or_crop(
                    sct[z],
                    hw,
                    -1000
                )

                inp_slices.append(
                    np.stack([a, b])
                )

                transforms.append(t)

            inputs = torch.from_numpy(
                np.stack(inp_slices)
            ).float()

            # ------------------------------------------------
            # INFERENCE
            # ------------------------------------------------

            pred_slices = []

            for start in range(
                0,
                len(inputs),
                bs
            ):

                batch = inputs[
                    start:start + bs
                ].to(device)

                pred = model(
                    batch
                )

                pred_slices.append(
                    pred.cpu().numpy()[:, 0]
                )

            # ------------------------------------------------
            # RESTORE ORIGINAL DIMENSIONS
            # ------------------------------------------------

            predictions = np.concatenate(
                pred_slices
            )

            err = np.stack(
                [
                    restore(s, t)
                    for s, t in zip(
                        predictions,
                        transforms
                    )
                ]
            )

            # ------------------------------------------------
            # APPLY MASK
            # ------------------------------------------------

            err *= mask

            # ------------------------------------------------
            # CORRECTED sCT
            # ------------------------------------------------

            corrected = (
                sct + err
            )

            # ------------------------------------------------
            # OUTPUT FILENAMES
            # ------------------------------------------------

            stem = (
                name[:-7]
                if name.endswith(".nii.gz")
                else Path(name).stem
            )

            # ------------------------------------------------
            # SAVE IMAGES
            # ------------------------------------------------

            for label, arr in (
                ("error", err),
                ("corrected", corrected)
            ):

                img = sitk.GetImageFromArray(
                    arr.astype(np.float32)
                )

                img.CopyInformation(
                    reference
                )

                output_path = (
                    dest
                    / f"{stem}_{label}.nii.gz"
                )

                sitk.WriteImage(
                    img,
                    str(output_path)
                )

            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            valid = mask > 0

            if valid.any():

                mae_before = float(
                    np.abs(
                        ct[valid] - sct[valid]
                    ).mean()
                )

                mae_after = float(
                    np.abs(
                        ct[valid] - corrected[valid]
                    ).mean()
                )

                bias_before = float(
                    (
                        ct[valid] - sct[valid]
                    ).mean()
                )

                bias_after = float(
                    (
                        ct[valid] - corrected[valid]
                    ).mean()
                )

                rows.append(
                    {
                        "patient": pid,
                        "sct": name,
                        "mae_before": mae_before,
                        "mae_after": mae_after,
                        "bias_before": bias_before,
                        "bias_after": bias_after
                    }
                )

                print(
                    f"[METRICS] "
                    f"MAE: {mae_before:.2f} -> {mae_after:.2f} HU | "
                    f"Bias: {bias_before:.2f} -> {bias_after:.2f} HU",
                    flush=True
                )

    # ========================================================
    # SAVE TEST METRICS
    # ========================================================

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    metrics_path = (
        out_dir / "metrics.csv"
    )

    with metrics_path.open(
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "patient",
                "sct",
                "mae_before",
                "mae_after",
                "bias_before",
                "bias_after"
            ]
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print(
        f"\n[TEST COMPLETED] Metrics saved to: {metrics_path}",
        flush=True
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Patient-level synthetic CT "
            "signed-error prediction and correction."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to configuration file"
    )

    parser.add_argument(
        "--mode",
        choices=[
            "train",
            "test",
            "train-test"
        ],
        default="train-test"
    )

    parser.add_argument(
        "--checkpoint",
        help="Required in test mode"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # READ CONFIG
    # --------------------------------------------------------

    cfg = configparser.ConfigParser(
        inline_comment_prefixes=(";", "#")
    )

    if not cfg.read(
        args.config
    ):

        raise FileNotFoundError(
            args.config
        )

    # --------------------------------------------------------
    # DATA SPLITS
    # --------------------------------------------------------

    splits = Path(
        cfg["paths"]["splits_dir"]
    )

    train_ids = (
        read_ids(
            splits / "train.txt"
        )
        if args.mode != "test"
        else []
    )

    val_ids = (
        read_ids(
            splits / "validation.txt"
        )
        if args.mode != "test"
        else []
    )

    test_ids = (
        read_ids(
            splits / "testing.txt"
        )
        if args.mode != "train"
        else []
    )

    # --------------------------------------------------------
    # PATIENT LEAKAGE CHECK
    # --------------------------------------------------------

    if (
        set(train_ids) & set(val_ids)
        or set(train_ids) & set(test_ids)
        or set(val_ids) & set(test_ids)
    ):

        raise ValueError(
            "Patient leakage detected: "
            "train/validation/test overlap."
        )

    # --------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------

    out = Path(
        cfg["paths"]["output_dir"]
    )

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device_name = cfg.get(
        "hardware",
        "device",
        fallback="auto"
    ).strip().lower()

    if device_name == "auto":
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"

    try:
        device = torch.device(device_name)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(
            f"Invalid [hardware] device: {device_name!r}. "
            "Use auto, cpu, cuda, or cuda:N (for example cuda:0)."
        ) from exc

    if device.type not in {"cpu", "cuda"}:
        raise ValueError(
            "[hardware] device must be auto, cpu, cuda, or cuda:N."
        )

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device {device}, but CUDA is unavailable. "
                "Set [hardware] device = cpu or auto, or check your CUDA setup."
            )

        gpu_index = device.index
        if gpu_index is None:
            gpu_index = torch.cuda.current_device()

        gpu_count = torch.cuda.device_count()
        if gpu_index >= gpu_count:
            raise ValueError(
                f"Requested GPU index {gpu_index}, but only {gpu_count} "
                "CUDA device(s) are visible. GPU indices start at 0."
            )

    print(
        f"\n[DEVICE] Using: {device}",
        flush=True
    )

    if device.type == "cuda":

        print(
            f"[GPU] {torch.cuda.get_device_name(device)}",
            flush=True
        )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    if args.mode != "test":

        ckpt = train(
            cfg,
            device,
            train_ids,
            val_ids,
            out / "models"
        )

    else:

        if not args.checkpoint:

            parser.error(
                "--checkpoint is required in test mode"
            )

        ckpt = Path(
            args.checkpoint
        )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    if args.mode != "train":

        predict(
            cfg,
            device,
            ckpt,
            test_ids,
            out / "predictions"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()