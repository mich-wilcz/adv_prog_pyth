import cv2
import numpy as np

image = cv2.imread("guy.png")
output = image.copy()


# zadanie 1

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

mask = np.zeros_like(image)

for (x, y, w, h) in faces:
    center = (x + w//2, y + h//2)
    axes = (w//2, h//2)
    cv2.ellipse(mask, center, axes, 0, 0, 360, (255,255,255), -1)

result = cv2.bitwise_and(output, mask)

cv2.imshow("Tylko twarz", result)
cv2.waitKey(0)
cv2.destroyAllWindows()


# zadanie 2
image = cv2.imread("guy.png")
output = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = output[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        center = (ex + ew//2, ey + eh//2)
        radius = max(ew, eh)//2
        cv2.circle(roi_color, center, radius, (0,0,0), -1)

cv2.imshow("Zasłonięte oczy", output)
cv2.waitKey(0)
cv2.destroyAllWindows()


# zadanie 3
image = cv2.imread("guy.png")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])
mask1 = cv2.inRange(hsv, lower_red, upper_red)

lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([179, 255, 255])
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

mask = cv2.bitwise_or(mask1, mask2)

result = cv2.bitwise_and(image, image, mask=mask)

cv2.imshow("Tylko czerwony kolor", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

