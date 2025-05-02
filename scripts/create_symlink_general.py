import os
import pathlib

# Define the source and destination patterns
source_base = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/multi_illumination/least_square/rotate"
dest_base = "/ist/ist-share/vision/relight/datasets/multi_illumination/shadings/least_square/v3/rotate/shadings_marigold"

# Iterate over the directories from 000000 to 815000 in steps of 1000
scenes = sorted(os.listdir(source_base))
for dir_name in scenes:
    
    source_path = os.path.join(source_base, dir_name, "shading_exr_perspective_v3_order6_marigold")
    dest_path = os.path.join(dest_base, dir_name)
    
    # Ensure source exists
    if not os.path.exists(source_path):
        print(f"Skipping {source_path}, does not exist.")
        continue
    
    # Ensure the destination parent directory exists
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    # Create the symlink if it doesn't already exist
    try:
        if os.path.lexists(dest_path):  # Check if the symlink or file exists
            os.remove(dest_path)  # Remove it if it exists
        os.symlink(source_path, dest_path)
        print(f"Symlink created: {dest_path} -> {source_path}")
    except OSError as e:
        print(f"Failed to create symlink for {dir_name}: {e}")