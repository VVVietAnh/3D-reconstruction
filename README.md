# 3D Reconstruction Evaluation Pipeline

This project implements a pipeline to evaluate the quality of 3D reconstruction from two different methods:
1. COLMAP only
2. COLMAP + Instant-NGP (with geometry extraction)

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