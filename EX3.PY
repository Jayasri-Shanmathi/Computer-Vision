import cv2
import matplotlib.pyplot as plt

image=cv2.imread("images.jpg")
gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sobel_x=cv2.Sobel(
    gray,
    cv2.CV_64F,
    1,
    0,
    ksize=3
)
sobel_y=cv2.Sobel(
    gray,
    cv2.CV_64F,
    0,
    1,
    ksize=3
)
sobel_x=cv2.convertScaleAbs(sobel_x)
sobel_y=cv2.convertScaleAbs(sobel_y)
sobel_combined=cv2.addWeighted(
    sobel_x,
    0.5,
    sobel_y,
    0.5,
    0
)
laplacian=cv2.Laplacian(
    gray,
    cv2.CV_64F
)
laplacian=cv2.convertScaleAbs(laplacian)
canny=cv2.Canny(
    gray,
    100,
    200
)
cv2.imwrite("sobel_x.jpg", sobel_x)
cv2.imwrite("sobel_y.jpg", sobel_y)
cv2.imwrite("sobel_combined.jpg", sobel_combined)
cv2.imwrite("laplacian.jpg", laplacian)
cv2.imwrite("canny.jpg", canny)
plt.figure(figsize=(8,6))
plt.subplot(2,3,1)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
plt.subplot(2,3,2)
plt.imshow(sobel_x, cmap="gray")
plt.title("Sobel X")
plt.axis("off")
plt.subplot(2, 3, 3)
plt.imshow(sobel_y, cmap="gray")
plt.title("Sobel Y")
plt.axis("off")
plt.subplot(2, 3, 4)
plt.imshow(sobel_combined, cmap="gray")
plt.title("Sobel Combined")
plt.axis("off")
plt.subplot(2, 3, 5)
plt.imshow(laplacian, cmap="gray")
plt.title("Laplacian")
plt.axis("off")
plt.subplot(2, 3, 6)
plt.imshow(canny, cmap="gray")
plt.title("Canny")
plt.axis("off")
plt.tight_layout()
plt.show()