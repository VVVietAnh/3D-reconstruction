#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import shutil
import json
import math
import numpy as np
import cv2
from pathlib import Path
import logging
import struct
from read_write_model import read_model, write_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Convert COLMAP output to NeRF format")
    parser.add_argument("--video_in", default="", help="Run ffmpeg first to convert a provided video file into a set of images")
    parser.add_argument("--video_fps", default=2, type=float, help="FPS for video extraction")
    parser.add_argument("--time_slice", default="", help="Time slice in format t1,t2 for video extraction")
    parser.add_argument("--run_colmap", action="store_true", help="Run COLMAP first on the image folder")
    parser.add_argument("--colmap_matcher", default="exhaustive", choices=["exhaustive","sequential","spatial","transitive","vocab_tree"], help="COLMAP matcher type")
    parser.add_argument("--colmap_db", default="colmap.db", help="COLMAP database filename")
    parser.add_argument("--colmap_camera_model", default="OPENCV", choices=["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "OPENCV_FISHEYE"], help="Camera model")
    parser.add_argument("--colmap_camera_params", default="", help="Camera parameters")
    parser.add_argument("--images", type=str, required=True, help="Path to input images")
    parser.add_argument("--out", type=str, required=True, help="Path to output JSON file")
    parser.add_argument("--aabb_scale", type=int, default=4, help="AABB scale factor")
    parser.add_argument("--skip_early", type=int, default=0, help="Skip this many images from the start")
    parser.add_argument("--keep_colmap_coords", action="store_true", help="Keep COLMAP's original coordinate system")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--mask_categories", nargs="*", type=str, default=[], help="Object categories to mask out")
    args = parser.parse_args()
    return args

def run_ffmpeg(args):
    """Convert video to images using ffmpeg."""
    try:
        if not os.path.isabs(args.images):
            args.images = os.path.join(os.path.dirname(args.video_in), args.images)

        images = args.images
        video = args.video_in
        fps = float(args.video_fps) or 1.0
        
        logger.info(f"Running ffmpeg with input video file={video}, output image folder={images}, fps={fps}")
        
        if not args.overwrite and (input(f"Warning! Folder '{images}' will be deleted/replaced. Continue? (Y/n)").lower().strip()+"y")[:1] != "y":
            sys.exit(1)
            
        try:
            shutil.rmtree(args.images)
        except:
            pass
            
        os.makedirs(args.images, exist_ok=True)

        time_slice_value = ""
        if args.time_slice:
            start, end = args.time_slice.split(",")
            time_slice_value = f",select='between(t,{start},{end})'"
            
        cmd = f"ffmpeg -i {video} -qscale:v 1 -qmin 1 -vf \"fps={fps}{time_slice_value}\" {images}/%04d.jpg"
        logger.info(f"Executing: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
        
    except Exception as e:
        logger.error(f"Error during ffmpeg processing: {e}")
        raise

def clean_up(output_path):
    """Remove existing database and reconstruction folders."""
    logger.info(f"Cleaning up existing files in {output_path}")
    db_path = os.path.join(output_path, "colmap.db")
    sparse_path = os.path.join(output_path, "sparse")
    
    if os.path.exists(db_path):
        logger.info(f"Removing existing database: {db_path}")
        os.remove(db_path)
    if os.path.exists(sparse_path):
        logger.info(f"Removing existing sparse folder: {sparse_path}")
        shutil.rmtree(sparse_path)

def run_colmap_commands(input_path, output_dir, colmap_matcher="exhaustive"):
    """Run COLMAP commands to create sparse reconstruction."""
    try:
        # Create output directory for COLMAP
        logger.info(f"Creating COLMAP output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        # Set database path in output directory
        db_path = os.path.join(output_dir, "colmap.db")
        sparse_path = os.path.join(output_dir, "sparse")
        
        # Clean up any existing files
        clean_up(output_dir)
        
        # Create sparse directory
        logger.info(f"Creating sparse directory: {sparse_path}")
        os.makedirs(sparse_path, exist_ok=True)
                
        # Run feature extraction
        logger.info("Running COLMAP feature extraction...")
        feature_extractor_cmd = (
            f"colmap feature_extractor --database_path {db_path} --image_path {input_path} "
            f"--SiftExtraction.max_num_features 8192 "
            f"--SiftExtraction.max_image_size 2000 "
            f"--SiftExtraction.estimate_affine_shape true "
            f"--SiftExtraction.domain_size_pooling true"
        )
        logger.info(f"Executing: {feature_extractor_cmd}")
        subprocess.run(feature_extractor_cmd, shell=True, check=True)
        logger.info("Feature extraction completed successfully")
        
        # Run matcher
        logger.info(f"Running COLMAP {colmap_matcher} matcher...")
        matcher_cmd = (
            f"colmap {colmap_matcher}_matcher --database_path {db_path} "
            f"--SiftMatching.max_ratio 0.85"
        )
        logger.info(f"Executing: {matcher_cmd}")
        subprocess.run(matcher_cmd, shell=True, check=True)
        logger.info("Feature matching completed successfully")
        
        # Run mapper
        logger.info("Running COLMAP mapper...")
        mapper_cmd = (
            f"colmap mapper --database_path {db_path} --image_path {input_path} "
            f"--output_path {sparse_path} "
            f"--Mapper.min_num_matches 15 "
            f"--Mapper.ba_global_max_num_iterations 50"
        )
        logger.info(f"Executing: {mapper_cmd}")
        subprocess.run(mapper_cmd, shell=True, check=True)
        logger.info("Sparse reconstruction completed successfully")

        # Convert binary files to text format
        logger.info("Converting binary files to text format...")
        model_converter_cmd = (
            f"colmap model_converter --input_path {os.path.join(sparse_path, '0')} "
            f"--output_path {os.path.join(sparse_path, '0')} "
            f"--output_type TXT"
        )
        logger.info(f"Executing: {model_converter_cmd}")
        subprocess.run(model_converter_cmd, shell=True, check=True)
        logger.info("Binary to text conversion completed successfully")
        
        # Return path to the first reconstruction
        return os.path.join(sparse_path, "0")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"COLMAP command failed with error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during COLMAP processing: {e}")
        raise

def qvec2rotmat(qvec):
    """Convert quaternion vector to rotation matrix."""
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]
    ])

def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # handle exception for the opposite direction input
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def closest_point_2_lines(oa, da, ob, db):
    """Returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel."""
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c)**2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa+ta*da+ob+tb*db) * 0.5, denom

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def sharpness(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm

def convert_to_nerf_format(colmap_path, aabb_scale, input_path, keep_colmap_coords=False):
    """Convert COLMAP output to NeRF format."""
    # Get absolute path of input directory
    input_path = os.path.abspath(input_path)
    
    # Read cameras.txt
    cameras = {}
    with open(os.path.join(colmap_path, "cameras.txt"), "r") as f:
        camera_angle_x = math.pi / 2
        for line in f:
            if line[0] == "#":
                continue
            els = line.split(" ")
            camera = {}
            camera_id = int(els[0])
            camera["w"] = float(els[2])
            camera["h"] = float(els[3])
            camera["fl_x"] = float(els[4])
            camera["fl_y"] = float(els[4])
            camera["k1"] = 0
            camera["k2"] = 0
            camera["k3"] = 0
            camera["k4"] = 0
            camera["p1"] = 0
            camera["p2"] = 0
            camera["cx"] = camera["w"] / 2
            camera["cy"] = camera["h"] / 2
            camera["is_fisheye"] = False
            if els[1] == "SIMPLE_PINHOLE":
                camera["cx"] = float(els[5])
                camera["cy"] = float(els[6])
            elif els[1] == "PINHOLE":
                camera["fl_y"] = float(els[5])
                camera["cx"] = float(els[6])
                camera["cy"] = float(els[7])
            elif els[1] == "SIMPLE_RADIAL":
                camera["cx"] = float(els[5])
                camera["cy"] = float(els[6])
                camera["k1"] = float(els[7])
            elif els[1] == "RADIAL":
                camera["cx"] = float(els[5])
                camera["cy"] = float(els[6])
                camera["k1"] = float(els[7])
                camera["k2"] = float(els[8])
            elif els[1] == "OPENCV":
                camera["fl_y"] = float(els[5])
                camera["cx"] = float(els[6])
                camera["cy"] = float(els[7])
                camera["k1"] = float(els[8])
                camera["k2"] = float(els[9])
                camera["p1"] = float(els[10])
                camera["p2"] = float(els[11])
            elif els[1] == "SIMPLE_RADIAL_FISHEYE":
                camera["is_fisheye"] = True
                camera["cx"] = float(els[5])
                camera["cy"] = float(els[6])
                camera["k1"] = float(els[7])
            elif els[1] == "RADIAL_FISHEYE":
                camera["is_fisheye"] = True
                camera["cx"] = float(els[5])
                camera["cy"] = float(els[6])
                camera["k1"] = float(els[7])
                camera["k2"] = float(els[8])
            elif els[1] == "OPENCV_FISHEYE":
                camera["is_fisheye"] = True
                camera["fl_y"] = float(els[5])
                camera["cx"] = float(els[6])
                camera["cy"] = float(els[7])
                camera["k1"] = float(els[8])
                camera["k2"] = float(els[9])
                camera["k3"] = float(els[10])
                camera["k4"] = float(els[11])
            else:
                logger.error(f"Unknown camera model: {els[1]}")
                continue
            
            camera["camera_angle_x"] = math.atan(camera["w"] / (camera["fl_x"] * 2)) * 2
            camera["camera_angle_y"] = math.atan(camera["h"] / (camera["fl_y"] * 2)) * 2
            camera["fovx"] = camera["camera_angle_x"] * 180 / math.pi
            camera["fovy"] = camera["camera_angle_y"] * 180 / math.pi
            
            logger.info(f"Camera {camera_id}:")
            logger.info(f"\tResolution: {camera['w']}x{camera['h']}")
            logger.info(f"\tCenter: {camera['cx']}, {camera['cy']}")
            logger.info(f"\tFocal: {camera['fl_x']}, {camera['fl_y']}")
            logger.info(f"\tFOV: {camera['fovx']}, {camera['fovy']}")
            logger.info(f"\tDistortion: k={camera['k1']}, {camera['k2']} p={camera['p1']}, {camera['p2']}")
            
            cameras[camera_id] = camera
    
    if len(cameras) == 0:
        logger.error("No cameras found!")
        sys.exit(1)
    
    # Read images.txt and create output
    with open(os.path.join(colmap_path, "images.txt"), "r") as f:
        i = 0
        bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
        
        if len(cameras) == 1:
            camera = next(iter(cameras.values()))
            out = {
                "camera_angle_x": camera["camera_angle_x"],
                "camera_angle_y": camera["camera_angle_y"],
                "fl_x": camera["fl_x"],
                "fl_y": camera["fl_y"],
                "k1": camera["k1"],
                "k2": camera["k2"],
                "k3": camera["k3"],
                "k4": camera["k4"],
                "p1": camera["p1"],
                "p2": camera["p2"],
                "is_fisheye": camera["is_fisheye"],
                "cx": camera["cx"],
                "cy": camera["cy"],
                "w": camera["w"],
                "h": camera["h"],
                "aabb_scale": aabb_scale,
                "frames": []
            }
        else:
            out = {
                "frames": [],
                "aabb_scale": aabb_scale
            }
        
        up = np.zeros(3)
        for line in f:
            line = line.strip()
            if line[0] == "#":
                continue
            i = i + 1
            if i % 2 == 1:
                elems = line.split(" ")
                image_rel = os.path.relpath(input_path)
                name = str(f"./{image_rel}/{'_'.join(elems[9:])}")
                b = sharpness(name)
                logger.info(f"{name} sharpness={b}")
                
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                R = qvec2rotmat(-qvec)
                t = tvec.reshape([3,1])
                m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
                c2w = np.linalg.inv(m)
                
                if not keep_colmap_coords:
                    c2w[0:3,2] *= -1
                    c2w[0:3,1] *= -1
                    c2w = c2w[[1,0,2,3],:]
                    c2w[2,:] *= -1
                    up += c2w[0:3,1]
                
                frame = {
                    "file_path": os.path.join(input_path, os.path.basename(name)),
                    "sharpness": b,
                    "transform_matrix": c2w
                }
                if len(cameras) != 1:
                    frame.update(cameras[int(elems[8])])
                out["frames"].append(frame)
    
    nframes = len(out["frames"])
    
    if keep_colmap_coords:
        flip_mat = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        for f in out["frames"]:
            f["transform_matrix"] = np.matmul(f["transform_matrix"], flip_mat)
    else:
        up = up / np.linalg.norm(up)
        logger.info(f"up vector was {up}")
        R = rotmat(up, [0,0,1])
        R = np.pad(R, [0,1])
        R[-1, -1] = 1
        
        for f in out["frames"]:
            f["transform_matrix"] = np.matmul(R, f["transform_matrix"])
        
        logger.info("computing center of attention...")
        totw = 0.0
        totp = np.array([0.0, 0.0, 0.0])
        for f in out["frames"]:
            mf = f["transform_matrix"][0:3,:]
            for g in out["frames"]:
                mg = g["transform_matrix"][0:3,:]
                p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
                if w > 0.00001:
                    totp += p*w
                    totw += w
        if totw > 0.0:
            totp /= totw
        logger.info(f"center of attention: {totp}")
        
        for f in out["frames"]:
            f["transform_matrix"][0:3,3] -= totp
        
        avglen = 0.
        for f in out["frames"]:
            avglen += np.linalg.norm(f["transform_matrix"][0:3,3])
        avglen /= nframes
        logger.info(f"avg camera distance from origin: {avglen}")
        
        for f in out["frames"]:
            f["transform_matrix"][0:3,3] *= 4.0 / avglen
    
    for f in out["frames"]:
        f["transform_matrix"] = f["transform_matrix"].tolist()
    
    logger.info(f"{nframes} frames")
    return out

def main():
    args = parse_args()
    
    logger.info(f"Starting COLMAP to NeRF conversion")
    logger.info(f"Input images: {args.images}")
    logger.info(f"Output JSON: {args.out}")
    logger.info(f"AABB scale: {args.aabb_scale}")
    
    # Check that we can save the output before we do a lot of work
    try:
        open(args.out, "a").close()
    except Exception as e:
        logger.error(f"Could not save transforms JSON to {args.out}: {e}")
        sys.exit(1)
    
    try:
        # Convert video to images if needed
        if args.video_in:
            run_ffmpeg(args)
        
        # Run COLMAP
        logger.info("Running COLMAP...")
        colmap_path = run_colmap_commands(args.images, os.path.join(os.path.dirname(args.out), "temp_colmap"), args.colmap_matcher)
        
        # Convert to NeRF format
        nerf_data = convert_to_nerf_format(colmap_path, args.aabb_scale, args.images, args.keep_colmap_coords)
        
        # Save output
        logger.info(f"Saving output to {args.out}")
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(nerf_data, f, indent=2)
            
        logger.info(f"Successfully converted COLMAP output to NeRF format. Output saved to {args.out}")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise

if __name__ == "__main__":
    main() 