import os 
import skimage 
import requests
from multiprocessing import Pool
import json
from datasets import load_dataset
from tqdm.auto import tqdm
import numpy as np
import requests
from io import BytesIO


def process_image_from_url(meta, save_directory="images"):
    try:
        idx, url = meta
        current_directory = f"{(idx // 1000 * 1000):06d}"
        output_dir = os.path.join(save_directory, current_directory,)
        save_path = os.path.join(output_dir, f"{idx:06d}.jpg")
        if os.path.exists(save_path):
            return None

        # Create the save directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir,exist_ok=True)

        
        # Download the image from the URL
        try:
            #image = skimage.io.imread(url)

            # Fetch the image with a timeout of 10 seconds
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Check if the request was successful

            # Read the image from the content of the response
            image = skimage.io.imread(BytesIO(response.content))

        except:
            return None
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Calculate the center crop to make the image square
        if height > width:
            crop_size = width
            start_row = (height - width) // 2
            start_col = 0
        else:
            crop_size = height
            start_row = 0
            start_col = (width - height) // 2
        
        cropped_image = image[start_row:start_row + crop_size, start_col:start_col + crop_size]

        # Resize the cropped image to 1024x1024
        resized_image = skimage.transform.resize(cropped_image, (1024, 1024), anti_aliasing=True)

        # convert to RGB if RGBA
        if len(cropped_image.shape) == 3 and cropped_image.shape[2] == 4:
            # Normalize the image to range [0, 1]
            rgba_image = cropped_image / 255.0
            alpha_channel = rgba_image[..., 3:4] 
            
            background_color = np.array([0, 0, 0]) / 255.0 # background in [0,1]



            # Perform alpha blending
            rgb_image = (rgba_image[..., :3] * alpha_channel + background_color * (1 - alpha_channel))
            cropped_image = skimage.img_as_ubyte(rgb_image)
        
        # Convert to ubyte (0-255) range and save the image
        skimage.io.imsave(save_path, skimage.img_as_ubyte(resized_image))
        
        return None
    except:
        return None

def main():
    ds = load_dataset("dclure/laion-aesthetics-12m-umap")
    # building list  
    process_list = []
    with open('hires_index.json','r') as f:
        accept_list = json.load(f)
        for idx, a in enumerate(tqdm(accept_list)):
            process_list.append((idx, ds['train'][a]['URL']))
        
    with Pool(32) as p:
        results = list(tqdm(p.imap(process_image_from_url, process_list), total=len(process_list)))


if __name__ == "__main__":
    main()
