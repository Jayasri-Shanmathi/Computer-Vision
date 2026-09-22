import cv2
import matplotlib.pyplot as plt

image=cv2.imread("images.jpg")
gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, global_threshold = cv2.threshold(
    gray, 127, 255, cv2.THRESH_BINARY
)
adaptive_mean=cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)
adaptive_gaussian=cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)
_, otsu=cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
cv2.imwrite("global_threshold.jpg", global_threshold)
cv2.imwrite("adaptive_mean.jpg", adaptive_mean)
cv2.imwrite("adaptive_gaussian.jpg", adaptive_gaussian)
cv2.imwrite("otsu_threshold.jpg", otsu)
plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
plt.subplot(2, 3, 2)
plt.imshow(global_threshold, cmap="gray")
plt.title("Global Thresholding")
plt.axis("off")
plt.subplot(2, 3, 3)
plt.imshow(adaptive_mean, cmap="gray")
plt.title("Adaptive Mean")
plt.axis("off")
plt.subplot(2, 3, 4)
plt.imshow(adaptive_gaussian, cmap="gray")
plt.title("Adaptive Gaussian")
plt.axis("off")
plt.subplot(2, 3, 5)
plt.imshow(otsu, cmap="gray")
plt.title("Otsu Thresholding")
plt.axis("off")
plt.tight_layout()
plt.show()