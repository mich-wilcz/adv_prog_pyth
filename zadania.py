import cv2
import numpy as np
import imutils

image = cv2.imread("cat.jpg")
cv2.imshow("Oryginalny obraz", image)
cv2.waitKey(0)

(h, w) = image.shape[:2]
center = (w // 2, h // 2)

# zadanie 1
M = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated_45 = cv2.warpAffine(image, M, (w, h))

cv2.imshow("Obrót 45°", rotated_45)
cv2.waitKey(0)

# zadanie 2
M = cv2.getRotationMatrix2D(center, -90, 1.0)
rotated_neg90 = cv2.warpAffine(image, M, (w, h))

cv2.imshow("Obrót -90°", rotated_neg90)
cv2.waitKey(0)

# zadanie 3
corner = (0, 0)
M = cv2.getRotationMatrix2D(corner, 30, 1.0)
rotated_corner = cv2.warpAffine(image, M, (w, h))

cv2.imshow("Obrót 30° od rogu", rotated_corner)
cv2.waitKey(0)

# zadanie 4
try:
    angle = float(input("Podaj kąt obrotu: "))
except ValueError:
    print("Niepoprawny kąt.")
    exit()

M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated_custom = cv2.warpAffine(image, M, (w, h))

cv2.imshow(f"Obrót o {angle}°", rotated_custom)
cv2.waitKey(0)

# zadanie 5
rotated_180 = imutils.rotate(image, 180)

cv2.imshow("imutils.rotate 180°", rotated_180)
cv2.waitKey(0)

# zadanie 6
rotated_bound = imutils.rotate_bound(image, -33)

cv2.imshow("rotate_bound -33°", rotated_bound)
cv2.waitKey(0)

# zadanie 7
M = cv2.getRotationMatrix2D(center, 60, 1.0)
rotated_cv = cv2.warpAffine(image, M, (w, h))
rotated_imutils = imutils.rotate(image, 60)

cv2.imshow("warpAffine 60°", rotated_cv)
cv2.imshow("imutils.rotate 60°", rotated_imutils)
cv2.waitKey(0)

# zadanie 8
img_seq = image.copy()
for _ in range(3):
    M = cv2.getRotationMatrix2D(center, 30, 1.0)
    img_seq = cv2.warpAffine(img_seq, M, (w, h))

M90 = cv2.getRotationMatrix2D(center, 90, 1.0)
rotated_90 = cv2.warpAffine(image, M90, (w, h))

cv2.imshow("3 x 30°", img_seq)
cv2.imshow("1 x 90°", rotated_90)
cv2.waitKey(0)

# zadanie 9
M = cv2.getRotationMatrix2D(center, 75, 1.0)
rotated_75 = cv2.warpAffine(image, M, (w, h))
cv2.imwrite("rotated_output.jpg", rotated_75)

# zadanie 10
for angle in range(0, 361, 15):
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    cv2.imshow(f"Obrót {angle}°", rotated)
    cv2.waitKey(500)
