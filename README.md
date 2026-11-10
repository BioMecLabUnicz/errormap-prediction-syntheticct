# Synthetic CT error prediction and correction

This code trains a network to predict a signed error map from MRI and synthetic CT (sCT) images, then uses that map to correct the sCT:

```text
Target error = reference CT - sCT
Corrected sCT = sCT + predicted error
```

**Each patient can have one sCT or multiple sCTs.** Multiple reconstructions, such as those generated from different checkpoints of an MRI-to-sCT model, are optional. The script uses every sCT matching the configured filename pattern. You do not need the model or checkpoints used to generate those input images.

## 1. Install dependencies

In your Python environment:

```bash
python -m pip install torch monai numpy SimpleITK tensorboard
```

## 2. Prepare the data

Organize your images with one folder per patient:

```text
data/
├── patient_001/
│   ├── mri.nii.gz
│   ├── ct.nii.gz
│   ├── mask.nii.gz
│   └── sCT.nii.gz
└── patient_002/
    ├── mri.nii.gz
    ├── ct.nii.gz
    ├── mask.nii.gz
    └── sCT.nii.gz
```

For multiple sCTs, place them in the same patient folder, for example `sCT_01.nii.gz` and `sCT_02.nii.gz`. Patients do not need to have the same number of sCTs. Set the actual filenames and sCT pattern in the configuration.

**Input requirements:**

- Supply scalar 3D images. MRI, reference CT, sCT, and mask must already be aligned and have matching size, spacing, origin, and direction. The code does not register or resample them.
- CT and sCT intensities must be in Hounsfield units (HU). Mask values greater than zero define the evaluation and correction region.
- Verify that SimpleITK arrays represent axial slices at `array[z, :, :]`.
- Reference CT is required in all current modes, including testing, to calculate the evaluation metrics.

MRI is min–max normalized per patient. Slices are centre padded or cropped to the configured height and width; choose dimensions that retain the anatomy of interest. Predicted corrections are applied only inside the mask, with zero correction in regions removed by cropping.

Create a `splits/` folder containing:

| File | Purpose |
| --- | --- |
| `train.txt` | Patients used to train the model. |
| `validation.txt` | Patients used to select the best model. |
| `testing.txt` | Patients used for final evaluation. |

Each text file must contain one patient folder name per line:

```text
patient_001
patient_002
```

Use distinct patients across the three files, with no duplicates. All sCTs belonging to a patient stay in that patient's split. Split files are prepared by the user; the script does not generate them.

## 3. Configure the experiment

Copy the example configuration:

```bash
cp config.example.ini config.ini
```

Edit `config.ini`, starting with paths and image filenames. Relative paths are resolved from the directory where you run the command. Do not wrap INI values in quotes.

| Section | Parameter | Description |
| --- | --- | --- |
| `paths` | `data_dir` | Root folder containing patient images. |
| `paths` | `splits_dir` | Folder containing the split text files. |
| `paths` | `output_dir` | Folder where results will be saved. Use a new folder per experiment to avoid overwriting results. |
| `paths` | `sct_dir` | Optional separate root for sCTs, also organized by patient. If omitted, uses `data_dir`. |
| `images` | `mri_filename` | MRI filename within each patient folder. |
| `images` | `ct_filename` | Reference CT filename. |
| `images` | `mask_filename` | Anatomical mask filename. |
| `images` | `sct_glob` | Input sCT filename or pattern. `sCT.nii.gz` selects a single named image; `sCT*.nii.gz` selects all matching images. Do not match corrected outputs. |
| `images` | `include_ideal_sct` | If `true`, adds reference CT as an extra, zero-error sCT during training, validation, and testing. Otherwise use `false`. |
| `hardware` | `device` | Execution device, for example `cuda:0`, `cuda:2`, or `cpu`. `auto` selects CUDA when available, otherwise CPU. |
| `model` | `backbone` | Encoder architecture; example: `efficientnet-b4`. |
| `model` | `pretrained` | Whether to load pretrained encoder weights. `true` may require a download. |
| `model` | `dropout` | Dropout probability; example: `0.3`. |
| `model` | `height` | Target slice height in pixels. |
| `model` | `width` | Target slice width in pixels. Use dimensions supported by the backbone; example: `256 × 256`. |
| `training` | `seed` | Random seed. |
| `training` | `learning_rate` | Initial optimizer learning rate. |
| `training` | `weight_decay` | AdamW weight decay; `0` disables it. |
| `training` | `l1_lambda` | Weight of L1 parameter regularization; `0` disables it. |
| `training` | `batch_size` | Number of slices per training/validation batch. |
| `training` | `num_workers` | DataLoader worker count; `0` uses the main process. |
| `training` | `max_epochs` | Maximum training epochs. The example uses `2` for a functional check; increase for training. |
| `scheduler` | `factor` | Learning-rate reduction multiplier; `0.5` halves it. Must be between `0` and `1`. |
| `scheduler` | `patience` | Validation plateau patience for learning-rate reduction, in epochs. |
| `early_stopping` | `patience` | Consecutive epochs without a lower validation loss before stopping. |
| `inference` | `batch_size` | Number of slices per prediction batch. |

## 4. Run the code

Run these commands from the repository folder.

**Train, validate, and test:**

```bash
python run.py --config config.ini --mode train-test
```

The script trains the model, saves the checkpoint with the lowest validation loss, then loads it to evaluate test patients. This is the default mode.

**Train and validate only:**

```bash
python run.py --config config.ini --mode train
```

Requires `train.txt` and `validation.txt`. No test predictions are produced.

**Test an already trained error prediction model:**

```bash
python run.py --config config.ini --mode test \
    --checkpoint results/smoke_test/models/best.pth
```

Requires `testing.txt` and a saved error prediction model checkpoint. Replace the checkpoint path with your own and use matching model settings and preprocessing. Training and validation lists are not loaded.

Here, `--checkpoint` refers to **the error prediction model saved by this script**, independently of how the input sCTs were generated. It is required only in `test` mode. Training modes start a new model; resuming training is not implemented.

Optionally save terminal messages and errors while displaying them:

```bash
python run.py --config config.ini --mode train-test 2>&1 | tee run_log.txt
```

## 5. Find the results

All results are saved under `output_dir`:

| Output | Contents |
| --- | --- |
| `models/best.pth` | Error prediction model with the lowest validation loss. |
| `logs/training_log.txt` | Training settings, epoch losses, learning rate, and final summary. |
| `logs/tensorboard/` | TensorBoard training and validation curves. |
| `predictions/<patient_id>/*_error.nii.gz` | Predicted signed error map for each input sCT. |
| `predictions/<patient_id>/*_corrected.nii.gz` | Corrected sCT for each input sCT. |
| `predictions/metrics.csv` | Masked MAE and signed bias before and after correction, per patient/sCT, in HU. |

Lower MAE means a closer match to the reference CT. Bias is defined as the mean of `CT - sCT` (or `CT - corrected_sCT`). Training and validation losses also include L1 regularization, so they differ from pure test MAE.

If `include_ideal_sct = true`, the metrics also contain ideal CT-derived entries named `__ideal_CT__.nii.gz`; analyse these separately from real sCTs. Test-only mode does not create a training log. The optional terminal log captures additional loading and test messages.

To view TensorBoard curves, use your output path:

```bash
tensorboard --logdir results/smoke_test/logs/tensorboard
```

**Memory:** training and validation samples are stored in RAM. For an initial check, use fewer patients or fewer sCTs if necessary. Reduce batch size if memory runs out during model execution.
