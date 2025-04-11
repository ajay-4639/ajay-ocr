import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image (grayscale)
img = cv2.imread(r'preprocessing\2141_001-pages-1_page-0001.jpg', cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded properly
if img is None:
    raise ValueError("Image not found or unable to load.")

# Create an empty array of the same shape to store the normalized image
norm_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

# Normalize the image to range 0-255
img_normalized = cv2.normalize(img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

# Save the normalized image
cv2.imwrite('preprocessing/normalized_image.jpg', img_normalized)

# Display images using matplotlib
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Normalized Image')
plt.imshow(img_normalized, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
