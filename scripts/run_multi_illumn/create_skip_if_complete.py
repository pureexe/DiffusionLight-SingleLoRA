import os 
import json 
from tqdm.auto import tqdm

def main():
    need_to_process = []
    for i in tqdm(range(816)):
        dirname = f"{(i * 1000):06d}"
        try:
            dataset_path = f"/ist/ist-share/vision/relight/datasets/laion-aesthetics-1024/images/{dirname}"
            output_path = f"/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024/{dirname}/raw"
            count_dataset_file = len(os.listdir(dataset_path))
            output_file = [ f for f in os.listdir(output_path) if f.endswith("_ev-00.png") ]
            count_output_file = len(output_file)
            print(f"Dataset: {dataset_path} has {count_dataset_file} files")
            print(f"Output: {output_path} has {count_output_file} files")
            if count_dataset_file != count_output_file:
                need_to_process.append(dirname)
        except:
            need_to_process.append(dirname)
    print("Total need to process: ", len(need_to_process))
    print("Need to process:")
    print(need_to_process)
    print("Total need to process: ", len(need_to_process))
    print("Total dataset: ", 816)   
    # save to json
    with open("need_to_process.json", "w") as f:
        json.dump(need_to_process, f, indent=4)
    
if __name__ == "__main__":
    main()