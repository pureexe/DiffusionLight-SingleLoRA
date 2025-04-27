import os 
import json
from datasets import load_dataset
from tqdm.auto import tqdm

def main():
    with open('../dataset_download/hires_index.json','r') as f:
        accept_list = json.load(f)
        print(len(accept_list))
    
    ds = load_dataset("dclure/laion-aesthetics-12m-umap")
    # building list  
    prompts = {}
    for idx, a in enumerate(tqdm(accept_list[:1000])):
        folder_id = idx // 1000
        filename = f"{folder_id*1000:06d}/{idx:06d}"
        prompts[filename] = ds['train'][a]['TEXT']
    
    with open('prompts_1k.json','w') as f:
        json.dump(prompts, f, indent=4)
    os.chmod('prompts_1k.json', 0o777)

if __name__ == "__main__":
    main()
