import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Load the image
image_path = os.path.expanduser("~/Desktop/MV/assignment2/fourierspectrum.pgm")
image = Image.open(image_path).convert("L")
image = np.array(image, dtype=np.float32) s

# Log Transformation
c_log = 255 / np.log(1 + np.max(image))
log_transformed = (c_log * np.log(1 + image))
log_transformed = np.uint8(np.clip(log_transformed, 0, 255))

# Power-Law (Gamma) Transformation
gamma = 0.5  # Adjust gamma as needed
c_gamma = 255 / np.power(np.max(image), gamma)
gamma_transformed = (c_gamma * np.power(image, gamma))
gamma_transformed = np.uint8(np.clip(gamma_transformed, 0, 255))

# Histogram Equalization 
def histogram_equalization(img):
    hist, bins = np.histogram(img.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()  # Cumulative distribution function
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())  # Normalize CDF
    equalized_img = np.interp(img.flatten(), bins[:-1], cdf_normalized).reshape(img.shape)
    return np.uint8(equalized_img)

equalized_image = histogram_equalization(image)

# Compute Mean and Standard Deviation
def compute_stats(img):
    return np.mean(img), np.std(img)

stats = {
    "Original": compute_stats(image),
    "Log Transformed": compute_stats(log_transformed),
    "Gamma Transformed": compute_stats(gamma_transformed),
    "Histogram Equalized": compute_stats(equalized_image),
}

# Plot the results
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Original Image and Histogram
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title("Original Image")
axes[1, 0].hist(image.ravel(), bins=256, range=(0, 256))
axes[1, 0].set_title("Histogram (Original)")

# Log Transformed Image and Histogram
axes[0, 1].imshow(log_transformed, cmap='gray')
axes[0, 1].set_title("Log Transformed")
axes[1, 1].hist(log_transformed.ravel(), bins=256, range=(0, 256))
axes[1, 1].set_title("Histogram (Log)")

# Gamma Transformed Image and Histogram
axes[0, 2].imshow(gamma_transformed, cmap='gray')
axes[0, 2].set_title("Gamma Transformed")
axes[1, 2].hist(gamma_transformed.ravel(), bins=256, range=(0, 256))
axes[1, 2].set_title("Histogram (Gamma)")

# Histogram Equalized Image and Histogram
axes[0, 3].imshow(equalized_image, cmap='gray')
axes[0, 3].set_title("Histogram Equalized")
axes[1, 3].hist(equalized_image.ravel(), bins=256, range=(0, 256))
axes[1, 3].set_title("Histogram (Equalized)")

plt.tight_layout()
plt.show()

#Standard deviation
for key, (mean, std) in stats.items():
    print(f"{key}: Mean = {mean:.2f}, Standard Deviation = {std:.2f}")