import numpy as np
import imageio.v2 as imageio  # for reading HDR or PNG envmap
import matplotlib.pyplot as plt
import math

def get_fov():
    focal_path = '/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/focal/dir_11_mip2.npy'
    focal_px = np.load(focal_path) # focal length in term of pxiel
    fov_rad = 2 * np.arctan2(512, 2*focal_px)
    return fov_rad



def get_chromeball_distance(fov, ball_ratio=0.25):
    # Get chroomeball distance (d) assume chromeball has 1 unit radius
    # @see https://imgur.com/DJMBhbI
    theta = math.atan(math.tan(fov / 2) * ball_ratio)
    inv_theta = np.pi / 2 - theta
    d = (math.sin(inv_theta) / math.tan(theta)) + math.cos(inv_theta)
    return d

def cartesian_to_spherical(cartesian_coordinates):
    """Converts Cartesian coordinates to spherical coordinates.

    Args:
        cartesian_coordinates: A NumPy array of shape [..., 3], where each row
        represents a Cartesian coordinate (x, y, z).

    Returns:
        A NumPy array of shape [..., 3], where each row represents a spherical
        coordinate (r, theta, phi).
    """

    x, y, z = cartesian_coordinates[..., 0], cartesian_coordinates[..., 1], cartesian_coordinates[..., 2]
    r = np.linalg.norm(cartesian_coordinates, axis=-1)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)
    return np.stack([r, theta, phi], axis=-1)

# Settings
width, height = 1024, 1024
# fov = 60  # degrees
# d = 2.0  # sphere position at (-d, 0, 0)
fov = get_fov() * 180 / np.pi  # degrees
d = get_chromeball_distance(fov / 180 * np.pi)  # sphere position at (-d, 0, 0)


camera_pos = np.array([0, 0, 0])
sphere_center = np.array([-d, 0, 0])
sphere_radius = 1.0

# Load environment map (assume latitude-longitude format)
envmap = imageio.imread('/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/scripts/ball2persepctiveenvmap/output/envmap_grid/14n_copyroom1_128px_v18_alignconer_false/dir_11_mip2.png')  # shape (H, W, 3)
#envmap = imageio.imread('/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/scripts/ball2persepctiveenvmap/output/envmap_grid/14n_copyroom1_256px_v3/dir_11_mip2.png.normal.png')  # shape (H, W, 3)
env_h, env_w = envmap.shape[:2]

# Helper: normalize
def normalize(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)

# Helper: reflect
def reflect(I, N):
    return I - 2 * np.sum(I * N, axis=-1, keepdims=True) * N

# Helper: sample envmap given direction (latlong)
def sample_envmap(dir):
    x, y, z = dir[..., 0], dir[..., 1], dir[..., 2]
    theta = np.arccos(np.clip(z, -1, 1))       # [0, pi]
    phi = np.arctan2(y, x)                     # [-pi, pi]
    u = (phi + np.pi) / (2 * np.pi)
    v = theta / np.pi
    px = (u * env_w).astype(int) % env_w
    py = (v * env_h).astype(int) % env_h
    return envmap[py, px]

# Create image grid
aspect = width / height
x = np.linspace(-1, 1, width) * np.tan(np.radians(fov / 2)) * aspect
y = np.linspace(1, -1, height) * np.tan(np.radians(fov / 2))
px, py = np.meshgrid(x, y)
dirs = np.stack([-np.ones_like(px), px, py], axis=-1)
dirs = normalize(dirs)

# Ray-sphere intersection
oc = camera_pos - sphere_center
b = 2 * np.sum(dirs * oc, axis=-1)
c = np.sum(oc**2) - sphere_radius**2
discriminant = b**2 - 4 * c
hit_mask = discriminant > 0

# Initialize image
image = np.zeros((height, width, 3))

# Compute hit points

sqrt_disc = np.sqrt(discriminant[hit_mask])
t = (-b[hit_mask] - sqrt_disc) / 2
hit_pos = camera_pos + t[..., None] * dirs[hit_mask]
normals = normalize(hit_pos - sphere_center)
reflected = normalize(reflect(dirs[hit_mask], normals))

# Sample envmap at reflection direction
color = sample_envmap(reflected)
image[hit_mask] = color / 255.0  # normalize if 8-bit envmap

# Save output as PNG
output_image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
imageio.imwrite("chromeball_rendering_new_env.png", output_image)


# Save output as PNG
output_image_crop = output_image[384:640, 384:640]  # Crop to 256x256
imageio.imwrite("chromeball_rendering_new_env_crop.png", output_image_crop)


sqrt_disc = np.sqrt(discriminant)
t = (-b - sqrt_disc) / 2
hit_pos = camera_pos + t[..., None] * dirs
normals = normalize(hit_pos - sphere_center)


save_normal = normals.copy()
save_normal = ((normals + 1.0)) / 2.0 
save_normal = save_normal * hit_mask[..., None]
# a = cartesian_to_spherical(save_normal[384,512])
# b = cartesian_to_spherical(save_normal[639,512])
# theta_a = a[2]
# theta_b = b[2]
# print(theta_a * 180 / np.pi)
# print(theta_b * 180 / np.pi)
# print("differnet")
# print(((theta_b - theta_a) / 2) * 180 / np.pi)
# print("fov:", fov)
# theta = a[1]


# phi = a[2]
# print(theta * 180 / np.pi)
# print(phi * 180 / np.pi)
# exit()




save_normal = save_normal[384:640, 384:640]  # Crop to 256x256
save_normal = (np.clip(save_normal, 0, 1) * 255).astype(np.uint8)
imageio.imwrite("chromeball_rendering_new_env_normal_crop.png", save_normal)
