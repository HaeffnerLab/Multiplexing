"""
Author: Weiwei Wu
Date: 10/2023
Description: Fit Zernike Polynomials to PSF images.
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from scipy.ndimage import center_of_mass
from skimage.draw import disk
from scipy.special import binom
from scipy.ndimage import gaussian_filter

"""
inputs(modify here ONLY)

1. File path to TIFF image.
2. Corping dimension if needed. 
    - [x_min, x_max, y_min, y_max ]
    - O.W leave it as None
3. Sigma for Gaussian fitting.
4. minimum significant intensity (from 0 to 1)
5. manually select centroid [y, x]
"""
file_path = "pictures/real_img.tiff"
# corp_dim = [1500, 1800, 541, 814]
corp_dim = None
sigma = 0
min_intensity = 0.2
# manu_center = [129, 143]
manu_center = None


# Step 1: Load the Image, Convert to Grayscale, and Normalize
def load_and_preprocess_image(file, crop_area=None, sigma_fit=0):
    # Load the image
    image = io.imread(file)
    print("Original image shape:", image.shape)

    # Crop the image if needed
    if crop_area is not None:
        x_min, x_max, y_min, y_max = crop_area
        image = image[y_min:y_max, x_min:x_max]
        print("Image shape after cropping:", image.shape)

    # Convert RGBA to RGB if necessary
    # Reason to use grey scale: simplify computing complexity;
    # Different channels of a color image may have varying intensity distributions and contrast,
    # which can lead to inconsistencies in analysis.
    if image.shape[-1] == 4:
        image = image[..., :3]

    # Convert to grayscale if it's a color image
    if len(image.shape) == 3 and image.shape[2] in [3, 4]:  # Check if image has color channels
        image = color.rgb2gray(image)
    print("converted to grey scale")
    # Normalize the intensity values to be between 0 and 1
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    # Apply Gaussian filter to the image
    smoothed_image = gaussian_filter(image, sigma=sigma_fit)

    return smoothed_image


# Load and preprocess the image
image = load_and_preprocess_image(file_path, corp_dim, sigma)  # corp if needed: [x_min, x_max, y_min, y_max ]
if image.ndim == 3 and image.shape[2] == 2:
    # Handle this specific case, or raise an error
    print("Error: Image has an unexpected shape", image.shape)


#
# print("a working picture is below")
# file_path2 = "real_img.tiff"
# image2 = load_and_preprocess_image(file_path2)
# if image2.ndim == 3 and image2.shape[2] == 2:
#     # Handle this specific case, or raise an error
#     print("Error: Image has an unexpected shape", image2.shape)
# print(image2.ndim, image2.shape)


# Show the preprocessed image
# plt.imshow(image, cmap='gray')
# plt.title('Preprocessed Image')
# plt.axis('off')
# plt.show()

# Step 2: Finding the Center of the Light Spot
def find_light_spot_center(image):
    # Calculate the centroid of the intensity distribution
    centroid = center_of_mass(image)
    print("the coordinates of the centroid is: ", f"y = {centroid[0]}, x = {centroid[1]}")
    # return centroid
    return centroid


def find_enclosing_circle(image, center, min_intensity):
    '''
    This fucntion calculates the maximum radius of a circle that fully encloses
    the significant light spot in a grayscale image.

    This circle will later be normalized into unit circle.
    '''
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    cy, cx = center

    # Compute the distance of each pixel from the centroid
    r_squared = (x - cx) ** 2 + (y - cy) ** 2
    # print("r_squared shape:", r_squared.shape)

    # Find the maximum distance within the light spot
    significant_pixels = []
    for y_coord in range(len(image)):
        for x_coord in range(len(image[0])):
            if image[y_coord][x_coord] > min_intensity:  # need to adjust the threshold
                significant_pixels.append([y_coord, x_coord])
    r_squared_significant = []
    for y_coord, x_coord in significant_pixels:
        r_squared_significant.append(r_squared[y_coord][x_coord])
    max_radius_sq = np.max(r_squared_significant)
    # print("radius from each pixel:", r_squared[image>0.1].shape)
    max_radius = np.sqrt(max_radius_sq)

    return max_radius, y - cy, x - cx


# Find the center of the light spot
center_y, center_x = find_light_spot_center(image)
if manu_center is not None:
    center_y,center_x = manu_center
# Find the maximum radius of the enclosing circle
max_radius, ycy, xcx = find_enclosing_circle(image, (center_y, center_x), min_intensity)

# Sanity check about dimensions.
# print(image.shape)
# print(ycy.shape)
# print((xcx+ycy).shape)

# Show the image with the centroid marked
fig, ax = plt.subplots()
ax.imshow(image, cmap='gray')
ax.scatter([center_x], [center_y], c='red', marker='x')
enclosing_circle = plt.Circle((center_x, center_y), max_radius, color='red', fill=False)
ax.add_patch(enclosing_circle)
ax.set_title('PSF with Centroid and Enclosing Circle')
ax.axis('off')


# plt.show()


# Step 3: Implement Zernike Transform and find coefficients for Zernike Polynomials
def polar_coordinates(image, center, radius):
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    cy, cx = center
    # normalize the possible r values, making it between 0 and 1
    normalized_r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / radius
    theta = np.arctan2(y - cy, x - cx)
    # print(r.shape)
    # print(theta.shape)
    return normalized_r, theta


def zernike_radial(m, n, rho):
    # rho here is normalized.
    if (n - abs(m)) % 2 == 1 or rho.any() > 1:
        return np.zeros_like(rho)
    Z = np.zeros_like(rho)
    for k in range((n - abs(m)) // 2 + 1):
        coef = ((-1) ** k) * binom(n - k, k) * binom(n - 2 * k, (n - m) // 2 - k)
        Z += coef * rho ** (n - 2 * k)
    return Z


def zernike(m, n, normalized_rho, phi):
    R = zernike_radial(m, n, normalized_rho)
    # print(phi.shape)
    if m > 0:
        Z = R * np.cos(m * phi)
        # print(Z.shape)
    elif m < 0:
        Z = R * np.sin(-m * phi)
    else:
        Z = R
    return Z


def compute_zernike_coefficients(image, center, radius):
    '''
    rho and phi have the shape/dimension of the picture. They are 2D arrays.

    rho and phi span the circle of the max radius defined previously.
    These rho and phi values are also normalized(rho are between 0 and 1) and
    applied to the Z_mn circles(poly. patterns), making sure the sizes of the circles between
    the input image and Zernike polynomials are matched.
    '''
    normalized_rho, phi = polar_coordinates(image, center, radius)

    coefficients = []
    for n in range(5):
        for m in range(-n, n + 1, 2):
            Z = zernike(m, n, normalized_rho, phi)
            epsilon_m = 1
            if m == 0:
                epsilon_m = 2
            # image * Z: inner product
            Coeff = np.sum(image * Z) * (n + 1) * 2 / (epsilon_m * np.pi * radius ** 2)
            coefficients.append((m, n, Coeff))
    return coefficients


# Compute the Zernike coefficients
coefficients = compute_zernike_coefficients(image, (center_y, center_x), max_radius)

# Step 4: construct the diagram
zernike_coefficients = []
for m, n, C in coefficients:
    # Displaying the coefficients
    print(f"Z_{m}^{n}: {C:.4f}")
    zernike_coefficients.append(C)

# Corresponding Zernike modes and aberrations, Z_m^n
zernike_modes = ["Z_0^0 (Piston)",
                 "Z_-1^1 (Vertical Tilt)",
                 "Z_1^1 (Horizontal Tile)",
                 "Z_-2^2 (Oblique Astigmatism)",
                 "Z_0^2 (Defocus)",
                 "Z_2^2 (Vertical Astigmatism)",
                 "Z_-3^3 (Vertical Trefoil)",
                 "Z_-1^3 (Vertical Coma)",
                 "Z_1^3 (Horizontal Coma)",
                 "Z_3^3 (Oblique Trefoil)",
                 "Z_-4^4 (Quadrafoil)",
                 "Z_-2^4 (Secondary Astigmatism)",
                 "Z_0^4 (Primary Spherical Aberration)"
                 ]

# Create bar chart
plt.figure(figsize=(10, 5))
bars = plt.bar(np.arange(len(zernike_coefficients) - 2), zernike_coefficients[:-2], tick_label=zernike_modes)

# Add labels and title
plt.ylabel('Coefficient Value')
plt.title('Zernike Coefficients for Aberrations')
plt.xticks(rotation=45, ha='right')

for bar in bars:
    if bar.get_height() < 0:
        bar.set_color('tab:blue')
    else:
        bar.set_color('tab:blue')

# Reconstruct the light spot
def reconstruct_image(coefficients, center, radius, image_shape):
    y, x = np.ogrid[:image_shape[0], :image_shape[1]]
    cy, cx = center

    # Compute polar coordinates
    normalized_rho = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / radius
    phi = np.arctan2(y - cy, x - cx)

    # Initialize reconstructed image
    reconstructed_image = np.zeros_like(normalized_rho)

    # Sum up Zernike polynomials weighted by their coefficients
    for m, n, C in coefficients:
        Z = zernike(m, n, normalized_rho, phi)
        reconstructed_image += C * Z

    return reconstructed_image

# Reconstruct the image using the calculated Zernike coefficients
reconstructed_image = reconstruct_image(coefficients, (center_y, center_x), max_radius, image.shape)

# Show the original and reconstructed images side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(reconstructed_image, cmap='gray')
axes[1].set_title('Reconstructed Image')
axes[1].axis('off')

# Show plot
plt.tight_layout()
plt.show()
