import os
import pathlib

# Define the source and destination patterns
# source_base = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024"
# dest_base = "/ist/ist-share/vision/relight/datasets/laion-shading/v3/train/shadings_marigold_v2"

# source_base = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/multi_illumination/least_square/rotate"
# dest_base = "/ist/ist-share/vision/relight/datasets/laion-shading/v3/rotate/shadings_marigold_v2"

source_base = "/pure/t1/output/DiffusionLight-SingleLoRA/laion-aesthetics-1024"
dest_base = "/pure/t1/datasets/laion-shading/v4/train/shadings_marigold"
#dest_base = "/ist/ist-share/vision/relight/datasets/laion-shading/v3/train/shadings_marigold_v2"

# source_base = "/pure/t1/output/DiffusionLight-SingleLoRA/laion-aesthetics-1024"
# dest_base = "/ist/ist-share/vision/relight/datasets/laion-shading/v3/train/shadings_marigold_v2"

print("START LOOP")
# Iterate over the directories from 000000 to 815000 in steps of 1000
for i in range(0, 816000, 1000):
    dir_name = f"{i:06d}"
    source_path = os.path.join(source_base, dir_name, "shading_exr_perspective_v3_order6_marigold_v2")
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

print("DONE")