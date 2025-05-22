import cv2
import os
image = cv2.imread("cat.jpg")


# zadanie 1
path = "cat.jpg"
if not os.path.exists(path):
    print(f"Plik '{path}' nie istnieje.")
else:
    image = cv2.imread(path)
    if image is None:
        print("Nie wczytano obrazu.")
    else:
        cv2.imshow("Oryginalny obraz", image)

# zadanie 2
image_color = cv2.imread(path, cv2.IMREAD_COLOR)
if image_color is not None:
    print("Liczba kanałów:", image_color.shape[2])

# zadanie 3
image_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
if image_gray is not None:
    print("Liczba kanałów:", len(image_gray.shape))

# zadanie 4
output_path = "cat_gray.jpg"
cv2.imwrite(output_path, image_gray)
print(f"Obraz zapisany jako {output_path}")

# zadanie 5
cv2.imshow("Kolor", image_color)
cv2.imshow("Szarość", image_gray)
print("5. Naciśnij dowolny klawisz, aby zamknąć oba okna.")
cv2.waitKey(0)
cv2.destroyAllWindows()

# zadanie 6
cv2.namedWindow("Skalowalne okno", cv2.WINDOW_NORMAL)
cv2.imshow("Skalowalne okno", image_color)
cv2.waitKey(0)