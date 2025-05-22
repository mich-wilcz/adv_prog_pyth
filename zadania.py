import cv2
import time
import imutils

image = cv2.imread("cat.jpg")
cv2.imshow("Oryginalny obraz", image)
cv2.waitKey(0)

(h, w) = image.shape[:2]
center = (w // 2, h // 2)

# zadanie 1
resized_half = cv2.resize(image, center)
cv2.imshow("Zmniejszony o 50%", resized_half)
cv2.waitKey(0)

# zadanie 2
resized_2x = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
cv2.imshow("Powiększenie 2x", resized_2x)
cv2.waitKey(0)

# zadanie 3
resized_fixed = cv2.resize(image, (200, 300))
cv2.imshow("Rozmiar 200x300", resized_fixed)
cv2.waitKey(0)

# zadanie 4
methods = {
    "INTER_NEAREST": cv2.INTER_NEAREST,
    "INTER_LINEAR": cv2.INTER_LINEAR,
    "INTER_CUBIC": cv2.INTER_CUBIC,
    "INTER_LANCZOS4": cv2.INTER_LANCZOS4
}

for name, method in methods.items():
    resized = cv2.resize(image, (w * 3, h * 3), interpolation=method)
    cv2.imshow(f"{name}", resized)
    cv2.waitKey(500)

# zadanie 5
resized_width = imutils.resize(image, width=500)
cv2.imshow("Szerokość 500 (z zachowaniem proporcji)", resized_width)
cv2.waitKey(0)

# zadanie 6
resized_height = imutils.resize(image, height=400)
cv2.imshow("Wysokość 400 (z zachowaniem proporcji)", resized_height)
cv2.waitKey(0)

# zadanie 7
resized_down = cv2.resize(image, (w // 5, h // 5), interpolation=cv2.INTER_AREA)
cv2.imshow("Zmniejszenie 5x - INTER_AREA", resized_down)
cv2.waitKey(0)

# zadanie 8
scale = 4
up_cubic = cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
up_lanczos = cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)

cv2.imshow("4x INTER_CUBIC", up_cubic)
cv2.imshow("4x INTER_LANCZOS4", up_lanczos)
cv2.waitKey(0)

# zadanie 9
for scale in range(100, 301, 20):
    new_w = int(w * scale / 100)
    new_h = int(h * scale / 100)
    resized = cv2.resize(image, (new_w, new_h))
    cv2.imshow(f"Skala: {scale}%", resized)
    cv2.waitKey(500)

# zadanie 10
resized_800 = imutils.resize(image, width=800)
cv2.imwrite("resized_output.jpg", resized_800)
