import os  
import json 

TOTAL_FRAME = 816
SHADING_DIR = "/ist/ist-share/vision/relight/datasets/laion-shading/v2/train/shadings"
INDEX_PATH = "/ist/ist-share/vision/relight/datasets/laion-shading/v2/train/index/main_v1.json"
PROMPT_PATH = "/ist/ist-share/vision/relight/datasets/laion-shading/v2/train/prompts.json"
DEFAULT_PROMPT = "a photorealistic image"

def main():
    indexs = {
        "image_index": [],
        "envmap_index": []
    }
    prompts = {}
    for scene_id in range(TOTAL_FRAME):
        scene_name = f"{scene_id*1000:06d}"
        shading_dir = os.path.join(
            SHADING_DIR,
            scene_name
        )
        if not os.path.exists(shading_dir):
            continue
        files = sorted(os.listdir(shading_dir))
        for filename in files:
            name = f"{scene_name}/{filename}"
            indexs['image_index'].append(name)
            indexs['envmap_index'].append([name])
            prompts[name] = DEFAULT_PROMPT

    with open(INDEX_PATH, "w") as json_file:
        json.dump(indexs, json_file, indent=4)
    with open(PROMPT_PATH, "w") as json_file:
        json.dump(prompts, json_file, indent=4)
    print("TOTAL FILES: ", len(indexs['image_index']))

if __name__ == "__main__":
    main()