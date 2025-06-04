import os
import tarfile
from tqdm.auto import tqdm
# Your source directories
prompt_dir = "/ist/ist-share/vision/relight/datasets/multi_illumination/unused/prompt_least_txt"
image_dir = "/ist/ist-share/vision/relight/datasets/multi_illumination/shadings/least_square/v4/train/images"
shading_dir = "/ist/ist-share/vision/relight/datasets/multi_illumination/unused/shading_least_npz"

# Output directory
output_dir = "/ist/ist-share/vision/relight/datasets/multi_illumination/shadings/least_square/v5/train"
os.makedirs(output_dir, exist_ok=True)

files = sorted([a.replace('.txt', '') for a in os.listdir(prompt_dir)])

num_samples = len(files)
samples_per_shard = 1000
num_shards = (num_samples + samples_per_shard - 1) // samples_per_shard


for shard_id in range(num_shards):
    print("CREATING... SHARD: ",shard_id)
    start_idx = shard_id * samples_per_shard
    end_idx = min((shard_id + 1) * samples_per_shard, num_samples)
    shard_name = os.path.join(output_dir, f"train-{shard_id:04d}.tar")

    with tarfile.open(shard_name, "w") as tar:
        for idx in tqdm(range(start_idx, end_idx)):
            base = files[idx]
            base_split= base.split('+')
            scene_name = base_split[0]
            image_name = base_split[1]
            txt_path = os.path.join(prompt_dir, f"{base}.txt")
            jpg_path = os.path.join(image_dir, scene_name, f"{image_name}.png")
            npz_path = os.path.join(shading_dir, f"{base}.npz")

            tar.add(txt_path, arcname=f"{base}.txt")
            tar.add(jpg_path, arcname=f"{base}.png")
            tar.add(npz_path, arcname=f"{base}.npz")
    
    print(f"Created {shard_name} with samples {start_idx}–{end_idx-1}")