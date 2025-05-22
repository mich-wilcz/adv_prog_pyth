import cv2
import numpy as np
import imutils

image = cv2.imread("cat.jpg")

cv2.imshow("Oryginalny obraz", image)
cv2.waitKey(0)

# zadanie 1

rows, cols = image.shape[:2]
M = np.float32([[1, 0, 30], [0, 1, 40]])
shifted = cv2.warpAffine(image, M, (cols, rows))

cv2.imshow("Przesunięcie prawo 30, dół 40", shifted)
cv2.waitKey(0)

# zadanie 2
M2 = np.float32([[1, 0, -20], [0, 1, -50]])
shifted_up_left = cv2.warpAffine(image, M2, (cols, rows))

cv2.imshow("Przesunięcie lewo 20, góra 50", shifted_up_left)
cv2.waitKey(0)

# zadanie 3
big_shift = np.float32([[1, 0, cols//2 + 50], [0, 1, rows//2 + 50]])
large_shifted = cv2.warpAffine(image, big_shift, (cols, rows))

cv2.imshow("Duże przesunięcie", large_shifted)
cv2.waitKey(0)

# zadanie 4
translated_imutils = imutils.translate(image, 100, 50)

cv2.imshow("imutils.translate", translated_imutils)
cv2.waitKey(0)

# zadanie 5
try:
    tx = int(input("Podaj przesunięcie w poziomie (tx): "))
    ty = int(input("Podaj przesunięcie w pionie (ty): "))
except ValueError:
    print("Niepoprawne dane wejściowe.")
    exit()

M_user = np.float32([[1, 0, tx], [0, 1, ty]])
shifted_user = cv2.warpAffine(image, M_user, (cols, rows))

cv2.imshow(f"Przesunięcie użytkownika tx={tx}, ty={ty}", shifted_user)
cv2.waitKey(0)
cv2.destroyAllWindows()
