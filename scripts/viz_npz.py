import os
import numpy as np
from PIL import Image
from multiprocessing import Pool

def process_file(npz_path, output_dir):
    try:
        # Load the npz file
        data = np.load(npz_path)
        
        # Get the first key in the npz file
        first_key = list(data.keys())[0]
        image_data = data[first_key]
        
        # Check the shape of the loaded array
        if image_data.shape == (1024, 1024, 3):
            # Normalize the data from [-1, 1] to [0, 255]
            image_data = ((image_data + 1) * 127.5).astype(np.uint8)
            
            # Convert the numpy array to an image
            image = Image.fromarray(image_data)
            
            # Save the image as PNG
            output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(npz_path))[0]}.png")
            image.save(output_path)
            print(f"Saved {output_path}")
        else:
            print(f"Skipping {npz_path}, invalid shape: {image_data.shape}")
    except Exception as e:
        print(f"Error processing {npz_path}: {e}")

def process_npz(input_dir, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get the list of .npz files in the input directory
    npz_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".npz")]

    # Use a multiprocessing Pool to process the files in parallel
    with Pool(8) as pool:
        pool.starmap(process_file, [(npz_path, output_dir) for npz_path in npz_files])

if __name__ == "__main__":
    input_directory = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024/000000/normal"  # Replace with your input directory path
    output_directory = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024/000000/normal_marigold_viz"  # Replace with your output directory path
    
    process_npz(input_directory, output_directory)