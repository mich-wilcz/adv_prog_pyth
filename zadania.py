import cv2
import numpy as np

image = cv2.imread("cat.jpg")
B, G, R = cv2.split(image)

# zadanie 1
cv2.imshow("Blue Channel", B)
cv2.imshow("Green Channel", G)
cv2.imshow("Red Channel", R)

cv2.imwrite("cat_blue.jpg", B)
cv2.imwrite("cat_green.jpg", G)
cv2.imwrite("cat_red.jpg", R)

cv2.waitKey(0)
cv2.destroyAllWindows()

# zadanie 3
reordered = cv2.merge([R, B, G])
cv2.imshow("R, B, G Order", reordered)

R_zeroed = np.zeros_like(R)
image_no_red = cv2.merge([B, G, R_zeroed])
cv2.imshow("Without Red", image_no_red)

cv2.waitKey(0)
cv2.destroyAllWindows()

# zadanie 4
R_boosted = cv2.add(R, 80)
image_boosted = cv2.merge([B, G, R_boosted])
cv2.imshow("Boosted Red", image_boosted)

cv2.waitKey(0)
cv2.destroyAllWindows()

# zadanie 5
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 120, 70])
upper_red2 = np.array([179, 255, 255])

mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = cv2.bitwise_or(mask1, mask2)

h, s, v = cv2.split(hsv)
s = cv2.add(s, 50, dst=s, mask=mask)
hsv_modified = cv2.merge([h, s, v])
result = cv2.cvtColor(hsv_modified, cv2.COLOR_HSV2BGR)

cv2.imshow("Selective Red Boost", result)
cv2.waitKey(0)

# zadanie 6
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 120, 70])
upper_red2 = np.array([179, 255, 255])

mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = cv2.bitwise_or(mask1, mask2)

h, s, v = cv2.split(hsv)
s = cv2.add(s, 50, dst=s, mask=mask)
hsv_modified = cv2.merge([h, s, v])
result = cv2.cvtColor(hsv_modified, cv2.COLOR_HSV2BGR)

cv2.imshow("Selective Red Boost", result)
cv2.waitKey(0)
