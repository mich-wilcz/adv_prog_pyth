import cv2
import os
import numpy as np

image = cv2.imread("cat.jpg")

# zadanie 1
print("Piksel w (0,0) [BGR]:", image[0, 0])
print("Składowe koloru (R, G, B):", image[0, 0][2], image[0, 0][1], image[0, 0][0])


# zadanie 2
img_copy = image.copy()
height, width = image.shape[:2]

print("Przed zmianą:", img_copy[height - 1, width - 1])
img_copy[height - 1, width - 1] = [0, 0, 255]  # czerwony

cv2.imshow("Oryginalny", image)
cv2.imshow("Po zmianie", img_copy)
cv2.waitKey(0)


# zadanie 3
center_x = width // 2
center_y = height // 2
center_pixel = image[center_y, center_x]

print(f"Środek obrazu: ({center_x}, {center_y})")
print("Wartości RGB w środku:", center_pixel[2], center_pixel[1], center_pixel[0])

# zadanie 4
try:
    x, y = map(int, input("Podaj współrzędne (x y): ").split())
    if 0 <= x < width and 0 <= y < height:
        img_black = image.copy()
        img_black[y, x] = [0, 0, 0]
        print(f"Piksel w ({x},{y}) ustawiony na czarny.")
        cv2.imshow("4. Piksel ustawiony na czarny", img_black)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Współrzędne poza zakresem obrazu")
except ValueError:
    print("Wpisz dwie liczby całkowite")

# zadanie 5
half_h, half_w = height // 2, width // 2
img_copy[:half_h, :half_w] = (255, 0, 0)  # niebieski (BGR)
cv2.imshow("Lewa górna ćwiartka = niebieska", img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()

# zadanie 6
img_center = image.copy()
center_x, center_y = width // 2, height // 2
x1, y1 = center_x - 50, center_y - 50
x2, y2 = center_x + 50, center_y + 50
img_center[y1:y2, x1:x2] = (0, 0, 255)  # czerwony
cv2.imshow("Środek = czerwony kwadrat", img_center)
cv2.waitKey(0)
cv2.destroyAllWindows()

# zadanie 7
h_step, w_step = height // 3, width // 3
y_start, y_end = h_step, 2 * h_step
x_start, x_end = w_step, 2 * w_step
img_center_tile = image[y_start:y_end, x_start:x_end]
cv2.imshow("Środkowa z 9 części", img_center_tile)
cv2.waitKey(0)
cv2.destroyAllWindows()

# zadanie 8
img_row = image.copy()
if height > 100:
    img_row[100, :] = (0, 255, 0)  # zielony
    cv2.imshow("Wiersz 100 = zielony", img_row)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Obraz za mały na 100. wiersz")

# zadanie 9
img_square = image.copy()
img_square[50:100, 50:100] = (255, 255, 255)  # biały
cv2.imshow("Obszar 50–100 = biały", img_square)
cv2.waitKey(0)
cv2.destroyAllWindows()

# zadani 10
pixel1 = image[50, 50]
pixel2 = image[200, 200] if height > 200 and width > 200 else pixel1
diff = np.abs(pixel1 - pixel2)
print(f"R: {diff[2]}, G: {diff[1]}, B: {diff[0]}")

# zadanie 11
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)
brightest_pixel = image[max_loc[1], max_loc[0]]  # (y, x)
print(f"Pozycja: {max_loc}")
print(f"RGB: {brightest_pixel[2]}, {brightest_pixel[1]}, {brightest_pixel[0]}")

