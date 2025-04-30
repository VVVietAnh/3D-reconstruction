#!/bin/bash
cd /home/doan/instant-ngp
source .venv/bin/activate
python3 scripts/colmap2nerf.py --colmap_matcher exhaustive --run_colmap --aabb_scale 4 --images "$(pwd)"/data/$1 --out ./output.json --overwrite
python3 scripts/run.py --save_mesh "$(pwd)"/output.ply output.json
