import cv2
import numpy as np

image = cv2.imread("cat.jpg")

h, w = image.shape[:2]

# zadanie 1
numpy_added = image + 50

opencv_added = cv2.add(image, np.ones(image.shape, dtype="uint8") * 50)

cv2.imshow("NumPy +50", numpy_added)
cv2.imshow("OpenCV +50", opencv_added)
cv2.waitKey(0)

# zadanie 2
burn_np = image + 150
burn_cv = cv2.add(image, np.full(image.shape, 150, dtype=np.uint8))

cv2.imshow("NumPy +150 (przepalenie)", burn_np)
cv2.imshow("OpenCV +150 (zabezpieczenie)", burn_cv)
cv2.waitKey(0)

# zadanie 3
darker_np = image - 80

darker_cv = cv2.subtract(image, np.full(image.shape, 80, dtype=np.uint8))

cv2.imshow("NumPy -80", darker_np)
cv2.imshow("OpenCV -80", darker_cv)
cv2.waitKey(0)

# zadanie 4
filtered = image.copy()
filtered = filtered.astype(np.int16)

filtered[:, :, 2] += 30   # Red +30
filtered[:, :, 1] -= 20   # Green -20
filtered[:, :, 0] += 10   # Blue +10

filtered = np.clip(filtered, 0, 255).astype(np.uint8)

cv2.imshow("Instagram filtr", filtered)
cv2.waitKey(0)

# zadanie 5
image1 = cv2.imread("cat.jpg")
image2 = cv2.imread("rotated_output.jpg")

if image1 is None or image2 is None:
    print("Nie udało się wczytać jednego z obrazów.")
    exit()

diff = cv2.absdiff(image1, image2)

cv2.imshow("Różnica obrazów", diff)
cv2.waitKey(0)

