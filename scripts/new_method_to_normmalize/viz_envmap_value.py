import ezexr
import matplotlib.pyplot as plt 
import numpy as np 

IMAGE_PATH = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/multi_illumination/least_square/rotate/everett_kitchen6/envmap_perspective_v3_pred_order_100v3/dir_0_mip2.exr"

def main():
    image = ezexr.imread(IMAGE_PATH)

    #image = np.clip(image, 0, 1.0)    
    # convert to gray scale 
    gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]

    print("MIN VALUE: ", gray.min())
    print("MAX VALUE: ", gray.max())
    # Plot and save to PNG
    plt.figure(figsize=(12, 10))
    img = plt.imshow(gray, cmap='viridis')
    plt.colorbar(img, label='Value')
    plt.title('Normalized Grayscale Visualization')
    plt.axis('off')
    plt.savefig("viz_brightness.png", bbox_inches='tight', dpi=300)
    plt.close()

    


if __name__ == "__main__":
    main()