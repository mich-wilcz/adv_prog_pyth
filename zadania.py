import cv2
import imutils
import numpy as np

image = cv2.imread("cat.jpg")
cv2.imshow("Oryginalny obraz", image)
cv2.waitKey(0)

(h, w) = image.shape[:2]
center = (w // 2, h // 2)

# zadanie 1
horizontal_flip = cv2.flip(image, 1)
cv2.imshow("Odbicie poziome", horizontal_flip)
cv2.waitKey(0)

# zadanie 2
vertical_flip = cv2.flip(image, 0)
cv2.imshow("Odbicie pionowe", vertical_flip)
cv2.waitKey(0)

# zadanie 3
both_flip = cv2.flip(image, -1)
cv2.imshow("Odbicie względem obu osi", both_flip)
cv2.waitKey(0)

# zadanie 4
horizontal = cv2.flip(image, 1)
vertical = cv2.flip(image, 0)
both = cv2.flip(image, -1)

# Zmniejszamy wszystkie obrazy do tej samej wysokości (dla ładnego układu)
h, w = image.shape[:2]
image_small = cv2.resize(image, (w // 2, h // 2))
horizontal_small = cv2.resize(horizontal, (w // 2, h // 2))
vertical_small = cv2.resize(vertical, (w // 2, h // 2))
both_small = cv2.resize(both, (w // 2, h // 2))

top_row = np.hstack((image_small, horizontal_small))
bottom_row = np.hstack((vertical_small, both_small))
grid = np.vstack((top_row, bottom_row))

cv2.imshow("Porównanie: Oryginał | Poziome | Pionowe | Oba", grid)
cv2.waitKey(0)

# zadanie 5
(h, w) = image.shape[:2]
right_half = image[:, w//2:]
flipped_half = cv2.flip(right_half, 1)  # odbicie poziome

# Wklejamy z powrotem
image_copy = image.copy()
image_copy[:, w//2:] = flipped_half

cv2.imshow("Odbicie tylko prawej połowy", image_copy)
cv2.waitKey(0)

# zadanie 6
choice = input("Wybierz typ odbicia (0 - pionowe, 1 - poziome, -1 - oba): ")
try:
    flip_code = int(choice)
    if flip_code not in [-1, 0, 1]:
        raise ValueError
    flipped = cv2.flip(image, flip_code)
    cv2.imshow(f"Odbicie użytkownika (flipCode={flip_code})", flipped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except ValueError:
    print("Niepoprawna wartość. Wybierz 0, 1 lub -1.")


