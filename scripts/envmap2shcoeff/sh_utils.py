import numpy as np
import pyshtools


def get_shcoeff(image, Lmax=100):
    """
    @param image: image in HWC @param 1max: maximum of sh
    """
    output_coeff = []
    for c_id in range(image.shape[-1]):
        # Create a SHGrid object from the image
        grid = pyshtools.SHGrid.from_array(image[:,:,c_id], grid='GLQ')
        # Compute the spherical harmonic coefficients
        coeffs = grid.expand(normalization='4pi', csphase=1, lmax_calc=Lmax)
        coeffs = coeffs.to_array()
        output_coeff.append(coeffs[None])
    
    output_coeff = np.concatenate(output_coeff,axis=0)
    return output_coeff

def unfold_sh_coeff(flatted_coeff, max_sh_level=2):
    """
    flatten spherical harmonics coefficient to 3xC matrix
    #  array format [0_0, 1_-1, 1_0, 1_1, 2_-2, 2_-1, 2_0, 2_1, 2_2]
    """
    sh_coeff = np.zeros((3, 2, max_sh_level+1, max_sh_level+1))
    for i in range(3):
        c = 0
        for j in range(max_sh_level+1):
            for k in range(j, 0, -1):
                sh_coeff[i, 1, j, k] = flatted_coeff[i, c]
                c +=1
            for k in range(j+1):
                sh_coeff[i, 0, j, k] = flatted_coeff[i, c]
                c += 1
    return sh_coeff

def flatten_sh_coeff(sh_coeff, max_sh_level=2):
    """
    flatten spherical harmonics coefficient to 3xC matrix
    """
    flatted_coeff = np.zeros((3, (max_sh_level+1) ** 2))
    # we will put into array in the format of 
    # [0_0, 1_-1, 1_0, 1_1, 2_-2, 2_-1, 2_0, 2_1, 2_2]
    # where first number is the order and the second number is the position in order
    for i in range(3):
        c = 0
        for j in range(max_sh_level+1):
            for k in range(j, 0, -1):
                flatted_coeff[i, c] = sh_coeff[i, 1, j, k]
                c +=1
            for k in range(j+1):
                flatted_coeff[i, c] = sh_coeff[i, 0, j, k]
                c += 1
    return flatted_coeff

def compute_background(
        hfov, sh, lmax=6,
        image_width=512, show_entire_env_map=True
    ):
    # Generate random spherical harmonic coefficients
    loaded_coeff = sh
    require_unfold = (lmax == 2)
    require_unfold = False

    if require_unfold:
        loaded_coeff = unfold_sh_coeff(loaded_coeff) #shape [3,9]
    
    output_image = []
    for ch in (range(3)):
        coeffs = loaded_coeff[ch]
            
        # Create SHCoeffs class object from the coefficients
        sh_coeffs = pyshtools.SHCoeffs.from_array(coeffs, lmax=lmax, normalization='4pi', csphase=1)

        # Create a grid of latitudes and longitudes
        if show_entire_env_map:
            theta = np.linspace(np.pi / 2, -np.pi / 2, image_width)
            phi = np.linspace(0, np.pi * 2, 2*image_width)
        else:
            theta = np.linspace(hfov, -hfov, image_width) #vertical
            phi = np.linspace(-hfov, hfov, image_width) #horizontal

        lat, lon = np.meshgrid(theta, phi, indexing='ij')

        # Evaluate the spherical harmonics on the grid
        grid_data = sh_coeffs.expand(grid="GLQ", lat=lat, lon=lon, lmax_calc=lmax, degrees=False)
        output_image.append(grid_data[...,None])    

    output_image = np.concatenate(output_image,axis=-1)
    print(output_image.max())
    #output_image = np.clip(output_image, 0.0 ,1.0) # MY CODE PROBLEM
    return output_image


def rotate_sh_coeff_x_axis(sh_coeff, angle):
    """
    Rotates the SHRealCoeff along the x-axis.

    Parameters:
        sh_coeff (numpy.ndarray): Spherical harmonic coefficients of shape [c, 3, L+1, L+1].
        angle (float): Rotation angle in radians. Positive values rotate right, negative values rotate left.

    Returns:
        numpy.ndarray: Rotated SHRealCoeff with the same shape as input.
    """
    _, _, L_plus_1, _ = sh_coeff.shape

    # Initialize the rotated coefficients with the same shape
    rotated_coeff = np.zeros_like(sh_coeff)

    lmax = 2
    djpi2 = pyshtools.rotate.djpi2(lmax)


    # Rotate for each channel separately
    for channel in range(3):
        # Extract the SH coefficients for this channel
        coeff = sh_coeff[channel]

        # Perform the rotation using PySHTools
        rotated = pyshtools.rotate.SHRotateRealCoef(coeff, np.array([angle,0,0]), djpi2)
        rotated_coeff[channel] = rotated

    return rotated_coeff