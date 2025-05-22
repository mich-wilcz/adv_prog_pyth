import cv2
import numpy as np

# zadanie 1
canvas_size = (300, 300)
blank = np.zeros(canvas_size, dtype=np.uint8)
triangle = blank.copy()
pts = np.array([[50, 250], [150, 50], [250, 250]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv2.fillPoly(triangle, [pts], 255)
circle = blank.copy()
cv2.circle(circle, (150, 150), 80, 255, -1)
bitwise_and = cv2.bitwise_and(triangle, circle)
bitwise_or = cv2.bitwise_or(triangle, circle)
bitwise_xor = cv2.bitwise_xor(triangle, circle)
bitwise_not = cv2.bitwise_not(triangle)
cv2.imshow("Trójkąt", triangle)
cv2.imshow("Okrąg", circle)
cv2.imshow("AND", bitwise_and)
cv2.imshow("OR", bitwise_or)
cv2.imshow("XOR", bitwise_xor)
cv2.imshow("NOT (trójkąt)", bitwise_not)
cv2.waitKey(0)
cv2.destroyAllWindows()

# zadanie 2
# Wczytaj dwa podobne obrazy
image1 = cv2.imread("cat.jpg")
image2 = cv2.imread("rotated_output.jpg")

# Różnice – XOR
xor_diff = cv2.bitwise_xor(image1, image2)

cv2.imshow("Obraz 1", image1)
cv2.imshow("Obraz 2", image2)
cv2.imshow("Różnice (XOR)", xor_diff)
cv2.waitKey(0)
cv2.destroyAllWindows()

