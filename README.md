# 3D Reconstruction Pipeline

This project implements a complete pipeline for 3D reconstruction using COLMAP and Instant-NGP, with evaluation capabilities for the Tanks and Temples dataset.

## Features

- COLMAP-based 3D reconstruction
- Instant-NGP integration
- Comprehensive evaluation metrics:
  - PSNR, SSIM, LPIPS for image quality
  - Chamfer Distance, Hausdorff Distance for 3D reconstruction quality
- Support for Tanks and Temples dataset

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- COLMAP 3.8+
- Instant-NGP

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd Reconstruction_3D
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install COLMAP:
```bash
# Follow COLMAP installation instructions from https://colmap.github.io/install.html
```

4. Install Instant-NGP:
```bash
# Follow Instant-NGP installation instructions from https://github.com/NVlabs/instant-ngp
```

## Project Structure

```
Reconstruction_3D/
├── scripts/                  # Main Python scripts
│   ├── colmap2nerf.py       # COLMAP to NeRF conversion
│   ├── inference_colmap.py  # COLMAP inference
│   ├── inference_ins_ngp.py # Instant-NGP inference
│   ├── preprocessing.py     # Data preprocessing
│   └── read_write_model.py  # COLMAP model handling
├── scripts/eval/            # Evaluation scripts
│   ├── evaluation_colmap.py # COLMAP evaluation
│   ├── evaluation_nerf.py   # NeRF evaluation
│   ├── evaluation_ply.py    # Point cloud evaluation
│   └── prepare_data.py      # Data preparation
├── data/                    # Dataset directory
├── output/                  # Output directory
├── colmap/                  # COLMAP installation
└── instant-ngp/            # Instant-NGP installation
```

## Usage

### Data Preparation

1. Prepare the Tanks and Temples dataset:
```bash
python scripts/eval/prepare_data.py --input_dir data --output_dir data/processed
```

### 3D Reconstruction

1. Run COLMAP reconstruction:
```bash
python scripts/inference_colmap.py --input_dir data/processed/tanks_and_temples/[scene]
```

2. Convert COLMAP output to NeRF format:
```bash
python scripts/colmap2nerf.py --images data/processed/tanks_and_temples/[scene]/images --out data/processed/tanks_and_temples/[scene]/transforms.json
```

3. Run Instant-NGP:
```bash
python scripts/inference_ins_ngp.py --scene data/processed/tanks_and_temples/[scene]
```

### Evaluation

1. Evaluate COLMAP reconstruction:
```bash
python scripts/eval/evaluation_colmap.py --scene [scene_name]
```

2. Evaluate NeRF reconstruction:
```bash
python scripts/eval/evaluation_nerf.py --scene [scene_name]
```

3. Evaluate point clouds:
```bash
python scripts/eval/evaluation_ply.py --gt data/gt/[scene].ply --colmap output/[scene]/colmap.ply --nerf output/[scene]/nerf.obj
```

## Evaluation Metrics

- **Image Quality Metrics**:
  - PSNR (Peak Signal-to-Noise Ratio)
  - SSIM (Structural Similarity Index)
  - LPIPS (Learned Perceptual Image Patch Similarity)

- **3D Reconstruction Metrics**:
  - Chamfer Distance
  - Hausdorff Distance

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The MIT License is a permissive free software license originating at the Massachusetts Institute of Technology. It is a simple license that permits reuse within proprietary software provided all copies of the licensed software include a copy of the MIT License terms and the copyright notice.

Key points of the MIT License:
- Permission is granted to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software
- The software is provided "as is" without warranty of any kind
- The license and copyright notice must be included in all copies or substantial portions of the software

For more information, please visit: https://opensource.org/licenses/MIT

