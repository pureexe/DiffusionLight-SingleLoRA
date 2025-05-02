import os  
import json 

TOTAL_FRAME = 816
SHADING_DIR = "/ist/ist-share/vision/relight/datasets/laion-shading/v3/train/shadings_marigold"
INDEX_PATH = "/ist/ist-share/vision/relight/datasets/laion-shading/v3/train/index/150k_marigold.json"
PROMPT_PATH = "/ist/ist-share/vision/relight/datasets/laion-shading/v3/train/prompts.json"
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
            filename = filename.split(".")[0]
            name = f"{scene_name}/{filename}"
            indexs['image_index'].append(name)
            indexs['envmap_index'].append([name])
            prompts[name] = DEFAULT_PROMPT
    indexs['image_index'] = indexs['image_index'][:150000]
    indexs['envmap_index'] = indexs['envmap_index'][:150000]            
    with open(INDEX_PATH, "w") as json_file:
        json.dump(indexs, json_file, indent=4)
    #with open(PROMPT_PATH, "w") as json_file:
    #    json.dump(prompts, json_file, indent=4)
    print("TOTAL FILES: ", len(indexs['image_index']))

if __name__ == "__main__":
    main()