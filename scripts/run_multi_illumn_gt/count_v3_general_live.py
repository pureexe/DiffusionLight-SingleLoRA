import os
import time

TOTAL_SCENE = 816
TEMPLATE = "/pure/t1/output/DiffusionLight-LoRASwapping/multi_illumination_gt/train"
ENVMAP_NAME = "envmap_perspective_v3"

def count_files():
    count = 0
    scenes = os.listdir(TEMPLATE)
    for scene in scenes:
        image_dir = os.path.join(TEMPLATE, scene, ENVMAP_NAME)
        try:
            files = os.listdir(image_dir)
        except:
            continue
        exr_files = [f for f in files if f.endswith(".exr")]
        count += len(exr_files)
    return count

def main():
    counts = []

    # Print header once
    print(f"{'Date':>10} | {'Time':>8} | {'Total Files':>12} | {'Change':>7}")
    print("-" * 47)

    while True:
        start_time = time.time()

        now = time.localtime()
        date_str = time.strftime("%m/%d", now)
        time_str = time.strftime("%H:%M:%S", now)

        total = count_files()
        prev = counts[-1] if counts else 0
        change = total - prev

        print(f"{date_str:>10} | {time_str:>8} | {total:12d} | {change:7d}")

        counts.append(total)

        elapsed = time.time() - start_time
        sleep_time = max(0, 60 - elapsed)
        time.sleep(sleep_time)

if __name__ == "__main__":
    main()