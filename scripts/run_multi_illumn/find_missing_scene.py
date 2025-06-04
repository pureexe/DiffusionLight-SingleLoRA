import os
import json
from tqdm.auto import tqdm

# Given path
base_path = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/multi_illumination/real/train/"

# Output file
output_json = "scenes_with_few_or_no_square_files.json"

# List all scene directories
scenes = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

# List to collect scenes with less than 25 files or no 'square' directory
problem_scenes = []

# Iterate over scenes and check 'square' subdirectory
for scene in tqdm(scenes):
    square_dir = os.path.join(base_path, scene, "square")
    if not os.path.exists(square_dir) or not os.path.isdir(square_dir):
        print(f"{scene} has no 'square' directory")
        problem_scenes.append(scene)
    else:
        file_count = len([f for f in os.listdir(square_dir) if os.path.isfile(os.path.join(square_dir, f))])
        if file_count < 25:
            print(f"{scene} has only {file_count} files in 'square'")
            problem_scenes.append(scene)

# Write to JSON file
with open(output_json, 'w') as f:
    json.dump(problem_scenes, f, indent=4)

print(f"\nSaved {len(problem_scenes)} scenes to '{output_json}'")
