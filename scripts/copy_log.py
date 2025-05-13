import os
import shutil

def copy_log_files(source_dir, target_dir):
    """
    Copy all log files from source directory to target directory
    """
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Copy all log files
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('points3D.txt') or file.endswith('images.txt') or file.endswith('cameras.txt'):
                continue
            if file.endswith('.txt') or file.endswith('.json'):
                new_dir = root.replace(source_dir, target_dir)
                os.makedirs(new_dir, exist_ok=True)
                print('file: ', os.path.join(root, file))
                print('new_dir: ', os.path.join(new_dir, file))
                shutil.copy(os.path.join(root, file), os.path.join(new_dir, file))

if __name__ == "__main__":
    source_dir = "output"
    target_dir = "output_log"
    copy_log_files(source_dir, target_dir)