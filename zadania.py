import cv2
import numpy as np

image = cv2.imread("guy.png")

h, w = image.shape[:2]

# zadanie 1
roi = image[0:100, 0:100].copy()
cv2.imshow("Lewy górny róg", roi)
cv2.waitKey(0)

# zadanie 2
lower_half = image[h//2:, :].copy()
cv2.imshow("Dolna połowa", lower_half)
cv2.waitKey(0)

# zadanie 3
right_half = image[:, w//2:].copy()
cv2.imshow("Prawa połowa", right_half)
cv2.waitKey(0)

# zadanie 4
startX = int(input("startX: "))
endX = int(input("endX: "))
startY = int(input("startY: "))
endY = int(input("endY: "))

if 0 <= startX < endX <= w and 0 <= startY < endY <= h:
    roi = image[startY:endY, startX:endX].copy()
    cv2.imshow("Dynamiczny ROI", roi)
else:
    print("Współrzędne poza zakresem!")

cv2.waitKey(0)

# zadanie 5
face = image[100:300, 150:350].copy()
cv2.imshow("Kadrowana twarz", face)
cv2.waitKey(0)

# zadanie 6
fragment = image[50:150, 50:150].copy()
image[200:300, 200:300] = fragment
cv2.imshow("Kopiowanie i wklejanie", image)
cv2.waitKey(0)

# zadanie 7
h_step = h // 3
w_step = w // 3

for row in range(3):
    for col in range(3):
        part = image[row*h_step:(row+1)*h_step, col*w_step:(col+1)*w_step].copy()
        cv2.imshow(f"Część ({row},{col})", part)
        cv2.waitKey(300)

# zadanie 8
roi_width = 100
for x in range(0, w - roi_width, 10):
    roi = image[0:100, x:x+roi_width]
    cv2.imshow("Przesuwający się ROI", roi)
    if cv2.waitKey(100) & 0xFF == 27:  # ESC to stop
        break

# zadanie 9
cropped = image[0:300, 0:300]
cv2.imwrite("cropped_image.jpg", cropped)
print("Zapisano jako cropped_image.jpg")
