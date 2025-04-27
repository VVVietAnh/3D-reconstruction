# 3D Reconstruction Pipeline

This repository contains a pipeline for 3D reconstruction using COLMAP and NeRF. The pipeline supports both DTU and Tanks and Temples datasets.

## Data Preparation

The data preparation script (`scripts/data/prepare_data.py`) prepares the datasets for both COLMAP and NeRF pipelines. It creates a structured output directory for each scene with the following structure:

```
scene_name/
├── images/           # All original images
├── sparse/          # COLMAP sparse reconstruction
├── input/           # Input for NeRF training
│   ├── images/      # Training images
│   ├── sparse/      # COLMAP sparse reconstruction
│   └── dense/       # COLMAP dense reconstruction
└── test.txt         # List of test image names
```

### Usage

```bash
python scripts/data/prepare_data.py [options]
```

Options:
- `--input_dir`: Input directory containing raw datasets (default: 'data')
- `--output_dir`: Output directory for processed data (default: 'data/processed')
- `--dataset`: Dataset to prepare ('dtu', 'tanks_and_temples', or 'all')
- `--k`: Number of images to select from the original set (default: 70)
- `--h`: Train/test ratio (e.g., '1:7' means 1 test image for every 7 train images)

### Example

```bash
# Prepare all datasets with default settings (70 images, 1:7 train/test ratio)
python scripts/data/prepare_data.py

# Prepare only DTU dataset with 100 images and 1:4 train/test ratio
python scripts/data/prepare_data.py --dataset dtu --k 100 --h 1:4
```

## COLMAP Pipeline

The COLMAP pipeline (`scripts/colmap/run_colmap.py`) performs sparse and dense reconstruction using COLMAP. It supports different parameter configurations for testing.

### Usage

```bash
python scripts/colmap/run_colmap.py [options]
```

Options:
- `--input_dir`: Input directory containing prepared data
- `--output_dir`: Output directory for COLMAP results
- `--test_mode`: Run in test mode with different parameter configurations
- `--config`: Path to COLMAP configuration file

### Example

```bash
# Run COLMAP with default settings
python scripts/colmap/run_colmap.py

# Run COLMAP in test mode
python scripts/colmap/run_colmap.py --test_mode
```

## NeRF Pipeline

The NeRF pipeline uses the COLMAP results to train a Neural Radiance Field model. The training images and camera parameters are taken from the prepared data directory.

## Directory Structure

```
.
├── data/                    # Raw and processed data
│   ├── dtu/                 # DTU dataset
│   └── tanks_and_temples/   # Tanks and Temples dataset
├── scripts/
│   ├── data/                # Data preparation scripts
│   ├── colmap/              # COLMAP pipeline scripts
│   └── nerf/                # NeRF pipeline scripts
└── README.md
```

## Requirements

- Python 3.8+
- COLMAP
- PyTorch
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone this repository
2. Install COLMAP (follow instructions at https://colmap.github.io/install.html)
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Project Structure

```
.
├── data/                    # Input data directory
│   ├── dtu/                # DTU dataset
│   └── tanks_and_temples/  # Tanks and Temples dataset
├── colmap/                 # COLMAP reconstruction pipeline
├── instant_ngp/           # Instant-NGP training and geometry extraction
├── evaluation/            # Evaluation metrics and visualization
├── results/               # Output results
└── utils/                 # Utility scripts
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install COLMAP:
```bash
# Follow COLMAP installation instructions from: https://colmap.github.io/install.html
```

3. Install Instant-NGP:
```bash
# Follow Instant-NGP installation instructions from: https://github.com/NVlabs/instant-ngp
```

## Usage

1. Prepare data:
```bash
python scripts/prepare_data.py
```

2. Run COLMAP reconstruction:
```bash
python scripts/run_colmap.py
```

3. Run COLMAP + Instant-NGP pipeline:
```bash
# Train Instant-NGP model
python scripts/train_instant_ngp.py

# Extract geometry from trained model
python scripts/extract_geometry.py
```

4. Evaluate results:
```bash
python scripts/evaluate.py
```

## Evaluation Metrics

The pipeline evaluates the following metrics:

### 3D Reconstruction Metrics
- Chamfer Distance
- Hausdorff Distance
- F-score
- Point-to-Point Distance
- Normal Consistency

### Image Quality Metrics
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)

## Results

Results will be saved in the `results/` directory, including:
- Quantitative metrics
- Visual comparisons
- Heatmaps
- Rendered image comparisons 

## Data Preparation Output

When running `prepare_data.py`, the following directory structure is created:

```
data/processed/
├── dtu/                    # Nếu chọn dataset dtu hoặc all
|   ├──scene0/
│   │   ├── images/             # Chứa tất cả ảnh
│   │   ├── sparse/             # Dành cho kết quả COLMAP sparse reconstruction
│   │   ├── input/
│   │   │   ├── images/         # Chứa các ảnh input
│   │   │   ├── sparse/         # Dành cho kết quả COLMAP sparse reconstruction tập input
│   │   │   ├── dense/          # Dành cho kết quả COLMAP dense reconstruction tập input
│   │   ├── test.txt            # Chứa tên các ảnh trong bộ test
│   ├──...
│
└── tanks_and_temples/     # Nếu chọn dataset tanks_and_temples hoặc all
|   ├──scene0/
│   │   ├── images/             # Chứa tất cả ảnh
│   │   ├── sparse/             # Dành cho kết quả COLMAP sparse reconstruction
│   │   ├── input/
│   │   │   ├── images/         # Chứa các ảnh input
│   │   │   ├── sparse/         # Dành cho kết quả COLMAP sparse reconstruction tập input
│   │   │   ├── dense/          # Dành cho kết quả COLMAP dense reconstruction tập input
│   │   ├── test.txt            # Chứa tên các ảnh trong bộ test
│   ├──...

