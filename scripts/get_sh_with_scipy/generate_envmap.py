import numpy as np 
import pyshtools
from sh_utils import sample_from_sh_numpy, sample_from_sh, unfold_sh_coeff
import ezexr

NPY_PATH = "/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/multi_illumination/least_square/rotate/everett_kitchen2/shcoeff_perspective_v3_order100_main/dir_0_mip2.npy"

def compute_background(sh, lmax=2, image_width=512):
    # Generate random spherical harmonic coefficients
    loaded_coeff = sh
    
    output_image = []
    for ch in (range(3)):
        coeffs = loaded_coeff[ch]
            
        # Create SHCoeffs class object from the coefficients
        sh_coeffs = pyshtools.SHCoeffs.from_array(coeffs, lmax=lmax, normalization='4pi', csphase=1)

        # Create a grid of latitudes and longitudes
        theta = np.linspace(np.pi / 2, -np.pi / 2, image_width)
        phi = np.linspace(0, np.pi * 2, 2*image_width)


        lat, lon = np.meshgrid(theta, phi, indexing='ij')

        # Evaluate the spherical harmonics on the grid
        grid_data = sh_coeffs.expand(grid="GLQ", lat=lat, lon=lon, lmax_calc=lmax, degrees=False)
        output_image.append(grid_data[...,None])    

    output_image = np.concatenate(output_image,axis=-1)
    return output_image

def compute_background2(sh, lmax=2, image_width=256):
    # Generate random spherical harmonic coefficients
    loaded_coeff = sh
    
    theta = np.linspace(np.pi / 2, -np.pi / 2, image_width)
    phi = np.linspace(0, np.pi * 2, 2*image_width)
    lat, lon = np.meshgrid(theta, phi, indexing='ij')
    output_image = []
    grid_data = sample_from_sh_numpy(shcoeff=loaded_coeff, lmax=lmax, theta=lat, phi=lon)
    output_image = grid_data
    return output_image


def main(): 
    # load sh feature 
    shcoeff = np.load(NPY_PATH)
    shcoeff = unfold_sh_coeff(shcoeff, max_sh_level=100)
    #envmmap = compute_background(sh=shcoeff,lmax=100)
    envmap = compute_background2(sh=shcoeff,lmax=100)
    envmap = envmap[...,:3]
    ezexr.imwrite("envmap2.exr", envmap)



if __name__ == "__main__":
    main()