import cv2
import numpy as np

image = cv2.imread("guy.png")


h, w = image.shape[:2]
center = (w // 2, h // 2)
bottom_right = (w - 1, h - 1)

# zadanie 1
cv2.line(image, center, bottom_right, (255, 0, 0), 2)  # niebieska linia

cv2.imshow("Linia od środka do rogu", image)
cv2.waitKey(0)

# zadanie 2
img = np.zeros((400, 400, 3), dtype=np.uint8)
cv2.rectangle(img, (0, 0), (100, 50), (0, 255, 0), -1)
cv2.rectangle(img, (300, 350), (399, 399), (0, 0, 255), 3)

cv2.imshow("Prostokąty", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# zadanie 3
img = np.zeros((300, 300, 3), dtype=np.uint8)


cv2.circle(img, (40, 40), 40, (255, 0, 0), -1)
cv2.circle(img, (150, 150), 60, (0, 0, 255), -1)

cv2.imshow("Okręgi", img)
cv2.waitKey(0)

# zadanie 4
img = np.zeros((300, 300, 3), dtype=np.uint8)

# Centrum obrazu
center = (150, 150)


top_left = (center[0] - 50, center[1] - 50)
bottom_right = (center[0] + 50, center[1] + 50)
cv2.rectangle(img, top_left, bottom_right, (255, 255, 255), 2)
cv2.circle(img, center, 30, (0, 0, 255), -1)

cv2.imshow("Kwadrat z okręgiem", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# zadanie 5
img = np.zeros((500, 500, 3), dtype=np.uint8)
center = (250, 250)

for i in range(5):
    size = 20 * (i + 1)
    top_left = (center[0] - size // 2, center[1] - size // 2)
    bottom_right = (center[0] + size // 2, center[1] + size // 2)
    cv2.rectangle(img, top_left, bottom_right, (255, 255, 255), 1)

cv2.imshow("Kwadraty w pętli", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# zadanie 6
image = cv2.imread("guy.png")

# Zakładamy orientacyjne współrzędne:
eye_left = (120, 180)
eye_right = (200, 180)
mouth = (150, 280)
face_center = (160, 190)
face_radius = 90

cv2.circle(image, eye_left, 15, (0, 0, 255), -1)
cv2.circle(image, eye_right, 15, (0, 0, 255), -1)

cv2.rectangle(image, (140, 240), (190, 260), (0, 255, 0), -1)
cv2.circle(image, face_center, face_radius, (255, 0, 0), 2)

cv2.imshow("Zamazanie twarzy", image)
cv2.waitKey(0)