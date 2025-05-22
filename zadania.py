import cv2
import os

# zadanie 1
image = cv2.imread("tiles.jpg")
scale_width = 300
height = int(image.shape[0] * scale_width / image.shape[1])
resized = cv2.resize(image, (scale_width, height))

gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

threshold_values = [100, 140, 180]
thresholded_images = []
for t in threshold_values:
    ret, thresh = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
    thresholded_images.append((t, thresh))
    cv2.imshow(f'Threshold {t}', thresh)

best_thresh_val = 140
ret, best_thresh = cv2.threshold(gray, best_thresh_val, 255, cv2.THRESH_BINARY)


# zadanie 2
def draw_contours_mode(mode):
    img_contours = resized.copy()
    contours, hierarchy = cv2.findContours(best_thresh, mode, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_contours, contours, -1, (0, 0, 255), 2)
    return img_contours, contours, hierarchy

modes = [cv2.RETR_EXTERNAL, cv2.RETR_TREE, cv2.RETR_LIST]
mode_names = ['RETR_EXTERNAL', 'RETR_TREE', 'RETR_LIST']

for mode, name in zip(modes, mode_names):
    img_c, contours, hierarchy = draw_contours_mode(mode)
    cv2.imshow(f'Contours {name}', img_c)

img_contours, contours, hierarchy = draw_contours_mode(cv2.RETR_EXTERNAL)


# zadanie 3
def test_resolution_scale(scale_w):
    h = int(image.shape[0] * scale_w / image.shape[1])
    img_resized = cv2.resize(image, (scale_w, h))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, best_thresh_val, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f'Scale width {scale_w}px -> liczba konturów: {len(contours)}')
    return img_resized, contours


scales_to_test = [100, 200, 300, 400]
for s in scales_to_test:
    test_resolution_scale(s)

# zadanie 4
output_dir = "kostki_wyciecia"
os.makedirs(output_dir, exist_ok=True)

img_numbered = resized.copy()
font = cv2.FONT_HERSHEY_SIMPLEX

for i, cnt in enumerate(contours, start=1):
    # zadanie 5
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img_numbered, (x, y), (x + w, y + h), (0, 255, 0), 2)
    text = f'{w}x{h} px'
    cv2.putText(img_numbered, text, (x, y - 10), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    cx, cy = x + w // 2, y + h // 2
    cv2.putText(img_numbered, str(i), (cx, cy), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    kostka = resized[y:y + h, x:x + w]
    cv2.imwrite(f"{output_dir}/kostka_{i:02d}.png", kostka)

# zadanie 6
filtered_contours = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if 95 < area < 5000:
        filtered_contours.append(cnt)

print(f"Liczba konturów po filtracji: {len(filtered_contours)}")

# zadanie 7
widths = []
heights = []

for cnt in filtered_contours:
    x, y, w, h = cv2.boundingRect(cnt)
    widths.append(w)
    heights.append(h)

if widths and heights:
    avg_w = sum(widths) / len(widths)
    avg_h = sum(heights) / len(heights)
    min_w, max_w = min(widths), max(widths)
    min_h, max_h = min(heights), max(heights)
    print(f"Liczba wykrytych kostek: {len(filtered_contours)}")
    print(f"Średnia szerokość: {avg_w:.2f} px, średnia wysokość: {avg_h:.2f} px")
    print(f"Minimalna szerokość: {min_w} px, maksymalna szerokość: {max_w} px")
    print(f"Minimalna wysokość: {min_h} px, maksymalna wysokość: {max_h} px")
else:
    print("Brak kostek po filtracji.")

cv2.imshow('Wynik', img_numbered)
cv2.waitKey(0)
cv2.destroyAllWindows()
