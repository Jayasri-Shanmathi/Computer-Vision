import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

folder = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(folder, "images.jpg")
image = cv2.imread(image_path)
if image is None:
    print("Error: Image not found!")
    exit()
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mean_filter = cv2.blur(image, (5, 5))
gaussian_filter = cv2.GaussianBlur(image, (5, 5), 0)
median_filter = cv2.medianBlur(image, 5)
bilateral_filter = cv2.bilateralFilter(image, 9, 75, 75)
cv2.imwrite(
    os.path.join(folder, "mean_filter.jpg"),
    mean_filter
)
cv2.imwrite(
    os.path.join(folder, "gaussian_filter.jpg"),
    gaussian_filter
)
cv2.imwrite(
    os.path.join(folder, "median_filter.jpg"),
    median_filter
)
cv2.imwrite(
    os.path.join(folder, "bilateral_filter.jpg"),
    bilateral_filter
)
mean_rgb = cv2.cvtColor(mean_filter, cv2.COLOR_BGR2RGB)
gaussian_rgb = cv2.cvtColor(gaussian_filter, cv2.COLOR_BGR2RGB)
median_rgb = cv2.cvtColor(median_filter, cv2.COLOR_BGR2RGB)
bilateral_rgb = cv2.cvtColor(bilateral_filter, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")
plt.subplot(2, 3, 2)
plt.imshow(mean_rgb)
plt.title("Mean Filter")
plt.axis("off")
plt.subplot(2, 3, 3)
plt.imshow(gaussian_rgb)
plt.title("Gaussian Filter")
plt.axis("off")
plt.subplot(2, 3, 4)
plt.imshow(median_rgb)
plt.title("Median Filter")
plt.axis("off")
plt.subplot(2, 3, 5)
plt.imshow(bilateral_rgb)
plt.title("Bilateral Filter")
plt.axis("off")
plt.tight_layout()
plt.show()