import cv2
import numpy as np
import imutils
import os

# Zadanie 1
fanta = cv2.imread("fanta.jpg")
logo = cv2.imread("logo.png")

result = cv2.matchTemplate(fanta, logo, cv2.TM_CCOEFF_NORMED)
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)

(h_logo, w_logo) = logo.shape[:2]
top_left = maxLoc
bottom_right = (top_left[0] + w_logo, top_left[1] + h_logo)
fanta_detected = fanta.copy()
cv2.rectangle(fanta_detected, top_left, bottom_right, (0, 255, 0), 2)

cv2.imshow("Fanta", fanta_detected)

print("Zad.1 - Współrzędne:", top_left, "| Dopasowanie (maxVal):", maxVal)

# Zadanie 2
for angle in [30, 45]:
    rotated = imutils.rotate(fanta, angle)
    result_rot = cv2.matchTemplate(rotated, logo, cv2.TM_CCOEFF_NORMED)
    _, maxVal_rot, _, maxLoc_rot = cv2.minMaxLoc(result_rot)
    print(f"Zad.2 - Rotacja {angle}° -> Dopasowanie (maxVal):", maxVal_rot)
    if maxVal_rot > 0.6:  # Próg detekcji
        cv2.rectangle(rotated, maxLoc_rot, (maxLoc_rot[0] + w_logo, maxLoc_rot[1] + h_logo), (255, 0, 0), 2)
    cv2.imshow(f'Rotated {angle}', rotated)

# Zadanie 3
scaled = cv2.resize(fanta, None, fx=1.5, fy=1.5)
result_scaled = cv2.matchTemplate(scaled, logo, cv2.TM_CCOEFF_NORMED)
_, maxVal_scaled, _, maxLoc_scaled = cv2.minMaxLoc(result_scaled)
print("Zad.3 - Skalowanie 150% -> Dopasowanie (maxVal):", maxVal_scaled)

# Zadanie 4
methods = [
    cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED,
    cv2.TM_CCORR, cv2.TM_CCORR_NORMED,
    cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED
]
method_names = [
    "TM_CCOEFF", "TM_CCOEFF_NORMED",
    "TM_CCORR", "TM_CCORR_NORMED",
    "TM_SQDIFF", "TM_SQDIFF_NORMED"
]

for method, name in zip(methods, method_names):
    result = cv2.matchTemplate(fanta, logo, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    match_loc = min_loc if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] else max_loc
    print(f"Zad.4 - Metoda {name} -> Dopasowanie: {min_val if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] else max_val}")
    temp_img = fanta.copy()
    cv2.rectangle(temp_img, match_loc, (match_loc[0] + w_logo, match_loc[1] + h_logo), (255, 0, 255), 2)

# Zadanie 5
screenshot = cv2.imread("screenshot.png")
icon = cv2.imread("mi.jpg")
res = cv2.matchTemplate(screenshot, icon, cv2.TM_CCOEFF_NORMED)
_, max_val_icon, _, loc_icon = cv2.minMaxLoc(res)
cv2.rectangle(screenshot, loc_icon, (loc_icon[0]+icon.shape[1], loc_icon[1]+icon.shape[0]), (0, 0, 255), 2)
print("Zad.5 - Detekcja ikony -> maxVal:", max_val_icon)

# Zadanie 6
coins_img = cv2.imread("isaac.jpg")
coins_template = cv2.imread("coin.jpg")
res = cv2.matchTemplate(coins_img, coins_template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
locs = np.where(res >= threshold)
print(f"Zad.6 - Liczba potencjalnych dopasowań: {len(locs[0])}")

# Zadanie 7
for pt in zip(*locs[::-1]):
    cv2.rectangle(coins_img, pt, (pt[0]+coins_template.shape[1], pt[1]+coins_template.shape[0]), (0, 255, 255), 2)

cv2.imshow("Znalezione", coins_img)

# Zadanie 7
img_objects = cv2.imread("isaac.jpg")
gray = cv2.cvtColor(img_objects, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,39 , 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
template = cv2.imread("coin.jpg")

for i, cnt in enumerate(contours):
    x, y, w, h = cv2.boundingRect(cnt)
    roi = img_objects[y:y+h, x:x+w]
    if roi.shape[0] >= template.shape[0] and roi.shape[1] >= template.shape[1]:
        res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > 0.1 :
            cv2.rectangle(img_objects, (x, y), (x + w, y + h), (0, 255, 0), 4)
            print(f"Zad.7 - Obiekt {i+1} pasuje do wzorca! maxVal: {max_val:.2f}")

cv2.imshow("Wyniki", img_objects)
cv2.waitKey(0)
cv2.destroyAllWindows()
