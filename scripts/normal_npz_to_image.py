import numpy as np
import imageio
import argparse

def load_and_save_npz(npz_path, output_path):
    # Load the npz file
    data = np.load(npz_path)
    
    # Assume the first key contains the image
    key = list(data.keys())[0]
    img = data[key]  # Shape: [1024, 1024, 3], Range: [-1, 1]
    
    # Normalize to [0,1]
    img = (img + 1) / 2.0
    
    # Scale to [0,255] and convert to uint8
    img = (img * 255).astype(np.uint8)
    
    # Save as PNG
    imageio.imwrite(output_path, img)
    print(f"Saved image to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NPZ image to PNG.")
    parser.add_argument("-npz_path", type=str, default="/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/laion-aesthetics-1024/000000/normal/000000.npz", help="Path to the input .npz file")
    parser.add_argument("-output_path", type=str, default="normal.png", help="Path to save the output .png file")
    
    args = parser.parse_args()
    load_and_save_npz(args.npz_path, args.output_path)
