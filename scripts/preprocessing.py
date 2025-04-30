import cv2
import os
from pathlib import Path
from rembg import remove
import zipfile
import shutil

def extract_frames(video_path, output_dir, num_frames):
    """
    Extracts a specific number of frames from a video and saves them to a directory.

    Args:
        video_path (str or Path): Path to the input video file.
        output_dir (str or Path): Directory where frames will be saved.
        num_frames (int): Number of frames to extract from the video.

    Returns:
        str: Path to the output directory containing extracted frames.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // num_frames)  # Interval between frames
    frame_number = 0
    saved_frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret or saved_frame_number >= num_frames:
            break

        if frame_number % frame_interval == 0:
            frame_filename = f"frame_{saved_frame_number:04d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            saved_frame_number += 1

        frame_number += 1

    cap.release()
    print(f"Extracted {saved_frame_number} frames to directory: {output_dir}")
    return output_dir

def extract_zip(zip_file, output_dir):
    """
    Extracts all files from a ZIP archive into a specified directory.

    Args:
        zip_file (str or Path): Path to the ZIP file to extract.
        output_dir (str or Path): Path to the output directory where files will be extracted.

    Returns:
        str: Path to the output directory containing extracted files.
    """
    os.makedirs(output_dir, exist_ok=True)
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(output_dir)
    return output_dir

def process_images(input_dir, output_dir):
    """
    Processes images by removing backgrounds and saving them as PNG files.

    Args:
        input_dir (str or Path): Directory containing input images.
        output_dir (str or Path): Directory where processed images will be saved.

    Returns:
        str: Path to the output directory containing processed images.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for file in os.listdir(input_dir):
        input_path = os.path.join(input_dir, file)
        
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            try:
                with open(input_path, "rb") as input_file:
                    input_image = input_file.read()
                    output_image = remove(input_image)

                output_file_path = os.path.join(output_dir, f"{Path(file).stem}.png")
                
                with open(output_file_path, "wb") as output_file:
                    output_file.write(output_image)
                    
            except Exception as e:
                print(f"Error processing {file}: {e}")

    print(f"Processed images saved to: {output_dir}")
    return output_dir

def process_input(input_path, output_dir, num_frames=None):
    """
    Main function to process input (video, zip, or image folder) and remove backgrounds.

    Args:
        input_path (str or Path): Path to input (video, zip, or image folder).
        output_dir (str or Path): Directory where processed images will be saved.
        num_frames (int, optional): Number of frames to extract from video. Required if input is video.

    Returns:
        str: Path to the output directory containing processed images.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    
    # Create temporary directory for intermediate files
    temp_dir = output_dir / "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Handle different input types
        if input_path.suffix.lower() in ['.mp4', '.mov', '.avi']:
            if num_frames is None:
                raise ValueError("num_frames must be specified when input is a video")
            # Extract frames from video
            temp_input_dir = extract_frames(input_path, temp_dir / "frames", num_frames)
        elif input_path.suffix.lower() == '.zip':
            # Extract zip file
            temp_input_dir = extract_zip(input_path, temp_dir / "extracted")
        elif input_path.is_dir():
            # Use image directory directly
            temp_input_dir = input_path
        else:
            raise ValueError(f"Unsupported input type: {input_path}")
        
        # Process images (remove background)
        final_output_dir = process_images(temp_input_dir, output_dir)
        
        return final_output_dir
        
    finally:
        # Clean up temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    # Example usage
    input_path = "data/videos/xe_F1_khan.mp4"  # Can be video, zip, or image folder
    root_output_dir = "data/processed_images"

    output_dir_name = os.path.basename(input_path).split(".")[0]
    output_dir = os.path.join(root_output_dir, output_dir_name)
    num_frames = 70  # Only needed if input is video
    
    process_input(input_path, output_dir, num_frames)

