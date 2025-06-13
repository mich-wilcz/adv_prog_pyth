import re
import cv2
import numpy as np
import easyocr
import os
import time
from ultralytics import YOLO
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from paddleocr import PaddleOCR


MODEL_OUTPUT_DIR = 'my_ocr_model'

paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')

reader = easyocr.Reader(['en'], model_storage_directory=MODEL_OUTPUT_DIR, user_network_directory=MODEL_OUTPUT_DIR,
                        gpu=True)

PLATE_PATTERNS = [
    # Dwuliterowe wyróżniki
    r'^[A-Z]{2}\s*\d{5}$',                # XY 12345
    r'^[A-Z]{2}\s*\d{4}[A-Z]$',           # XY 1234A
    r'^[A-Z]{2}\s*\d{3}[A-Z]{2}$',        # XY 123AB
    r'^[A-Z]{2}\s*[1-9]\d?[A-Z]\d{3}$',   # XY 1A234
    r'^[A-Z]{2}\s*[1-9]\d?[A-Z]{2}\d{2}$',# XY 1AB23

    # Trzyliterowe wyróżniki
    r'^[A-Z]{3}\s*[A-Z]\d{3}$',           # XYZ A123
    r'^[A-Z]{3}\s*\d{2}[A-Z]{2}$',        # XYZ 12AC
    r'^[A-Z]{3}\s*[1-9][A-Z]\d{2}$',      # XYZ 1A23
    r'^[A-Z]{3}\s*\d{2}[A-Z][1-9]$',      # XYZ 12A3
    r'^[A-Z]{3}\s*[1-9][A-Z]{2}[1-9]$',   # XYZ 1AC2
    r'^[A-Z]{3}\s*[A-Z]{2}\d{2}$',        # XYZ AC12
    r'^[A-Z]{3}\s*\d{5}$',                # XYZ 12345
    r'^[A-Z]{3}\s*\d{4}[A-Z]$',           # XYZ 1234A
    r'^[A-Z]{3}\s*\d{3}[A-Z]{2}$',        # XYZ 123AC
    r'^[A-Z]{3}\s*[A-Z]\d{2}[A-Z]$',      # XYZ A12C
    r'^[A-Z]{3}\s*[A-Z][1-9][A-Z]{2}$',   # XYZ A1CE
    r'^[A-Z]{3}\s*\d{3}[A-Z]$',           # XYZ 123A (nieoficjalny)

    # Zmniejszone tablice
    r'^[A-Z]\s*\d{3}$',                   # X 123
    r'^[A-Z]\s*\d{2}[A-Z]$',              # X 12A
    r'^[A-Z]\s*[1-9][A-Z][1-9]$',         # X 1A2
    r'^[A-Z]\s*[A-Z]\d{2}$',              # X A12
    r'^[A-Z]\s*[1-9][A-Z]{2}$',           # X 1AC
    r'^[A-Z]\s*[A-Z]{2}[1-9]$',           # X AC1
    r'^[A-Z]\s*[A-Z][1-9][A-Z]$',         # X A1C
]



def find_best_plate_match(ocr_text: str) -> str:
    # Wyczyść tekst z dziwnych znaków
    clean = re.sub(r'[^A-Z0-9]', '', ocr_text.upper())
    candidates = set()

    # Szukaj dopasowań do każdego wzorca
    for pattern in PLATE_PATTERNS:
        for match in re.finditer(pattern, clean):
            candidates.add(match.group())


    if not candidates:
        result = ocr_text[:8]
    else:
        result = max(candidates, key=len)

    if len(result) > 4 and len(result) > 1 and result[1] == '0':
        result = result[:1] + 'O' + result[2:]

    return result

def fuzzy_match(a, b, threshold):
    return SequenceMatcher(None, a, b).ratio() >= threshold


def evaluate(results, acc=1):
    correct = 0
    for filename, (prediction, gt_plate) in results.items():
        pred_norm = clean_plate_text(prediction)
        gt_norm = clean_plate_text(gt_plate)
        if pred_norm == gt_norm or fuzzy_match(pred_norm, gt_norm, acc):
            correct += 1
    return correct / len(results) * 100 if results else 0


def clean_plate_text(text):
    text = str(text)
    text = text.replace(" ", "").upper()
    return re.sub(r'[^A-Z0-9]', '', text)


def prostuj_tablice(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            angle = (theta * 180 / np.pi) - 90
            if -45 < angle < 45:  # ignorujemy pionowe linie
                angles.append(angle)

        if angles:
            sredni_kat = np.median(angles)
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, sredni_kat, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return img


def przytnij_marginesy(img, margines_x=0, margines_y=0):
    h, w = img.shape[:2]
    return img[margines_y:h - margines_y, margines_x:w - margines_x]


def calculate_final_grade(accuracy_percent: float, processing_time_sec: float) -> float:
    if accuracy_percent < 60 or processing_time_sec > 60:
        return 2.0
    accuracy_norm = (accuracy_percent - 60) / 40
    time_norm = (60 - processing_time_sec) / 50
    score = 0.7 * accuracy_norm + 0.3 * time_norm
    grade = 2.0 + 3.0 * score
    return round(grade * 2) / 2


def odczytaj_z_obrazka(img):
    step = prostuj_tablice(img)

    img_resized = cv2.resize(step, (800, 200))

    # Konwersja do skali szarości
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Usuwanie szumu (mediana)
    denoised = cv2.medianBlur(gray, 9) # 13

    # Binaryzacja - adaptive threshold
    binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 19, 2)

    # Wyszukiwanie konturów
    contours, _ = cv2.findContours(255 - binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #cv2.imshow("binary 1", binary)
    #cv2.waitKey(0)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 80 < h < 200 and 25 < w < 120 and h > w and w*h > 80*30:
            boxes.append((x, y, w, h))
            #cv2.rectangle(binary, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #cv2.imshow("binary", binary)
    #cv2.waitKey(0)

    cropped = gray
    if boxes:
        x_min = min([x for x, y, w, h in boxes]) - 10
        y_min = min([y for x, y, w, h in boxes]) - 10
        x_max = max([x + w for x, y, w, h in boxes]) + 10
        y_max = max([y + h for x, y, w, h in boxes]) + 10

        # Ograniczenie do wymiarów obrazu
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(binary.shape[1], x_max), min(binary.shape[0], y_max)

        # Wycięcie prostokąta
        cropped = img_resized[y_min:y_max, x_min:x_max]

        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # Usuwanie szumu (mediana)
    #denoised = cv2.medianBlur(gray, 17)

    final_text = ""
    paddle_result = paddle_ocr.ocr(cropped, cls=True)
    for res in paddle_result:
        for line in res:
            text = line[1][0]
            score = line[1][1]
            print(f"{text} ({score:.2f})")
            if (score >= 0.65 and len(text) < 5) or (score >= 0.8 and len(text) >= 5):
                final_text += text


    final_text = clean_plate_text(final_text)
    print(f"PADDLE: {final_text}")
    #cv2.imshow("cropped", cropped)
    #cv2.waitKey(0)

    #result_ocr = reader.readtext(cropped)
    #final_text_custom_crr = ""
    #for bbox, text, prob in result_ocr:
    #    if (len(text) <= 3 and prob > 0.4) or (len(text) > 3 and prob > 0.12):
    #        final_text_custom_crr += clean_plate_text(text)
    #print(f"MY OCR: {final_text_custom_crr}")

    return find_best_plate_match(final_text)


def crop_plate_strip(img_to_crop, margin_ratio=0.08):
    h, w = img_to_crop.shape[:2]
    x_margin = int(w * margin_ratio)
    cropped = img_to_crop[:, x_margin:]  # ucinamy z lewej strony
    return cropped


def load_annotations(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    data = []
    for image in root.findall('image'):
        name = image.get('name')
        for box in image.findall('box'):
            xtl = int(float(box.get('xtl')))
            ytl = int(float(box.get('ytl')))
            xbr = int(float(box.get('xbr')))
            ybr = int(float(box.get('ybr')))
            label = ""
            for attr in box.findall('attribute'):
                if attr.get('name') == 'plate number':
                    label = attr.text
            data.append((name, (xtl, ytl, xbr, ybr), label))
    return data


def compute_iou(boxA, boxB):
    # box = (x1, y1, x2, y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    unionArea = boxAArea + boxBArea - interArea
    if unionArea == 0:
        return 0.0

    return interArea / unionArea


#image = cv2.imread('cuts/2.jpg')

#odczytaj_liczby_z_obrazka(image)


annotations = load_annotations('annotations.xml')
annotations_dict = {name: label for name, _, label in annotations}


results = {}
ANNOTATIONS_FILE = 'annotations.xml'
model = YOLO('runs/detect/plate_yolov8/weights/best.pt')
all_ious = []
times = []

with open("test_files.txt", "r") as f:
    test_files = [line.strip() for line in f]


start_time = time.time()
for i, filename in enumerate(test_files):
    img_start_time = time.time()
    img_stop_time = time.time()

    path = os.path.join("photos", filename)
    image = cv2.imread(path)
    if image is None:
        continue

    boxes = [box for name, box, _ in annotations if name == filename]
    if not boxes:
        continue

    gt_plate = annotations_dict.get(filename, "BRAK")

    #img_results = model(cv2.resize(image, (2*1280, 2*720)))
    img_results = model(image)
    plate_text = ""

    ious = []
    for box in img_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate_crop = image[y1:y2, x1:x2]

        aligned_plate = crop_plate_strip(plate_crop)

        # OCR
        plate_text = odczytaj_z_obrazka(aligned_plate)
        img_stop_time = time.time()

        for gt in boxes:
            iou = compute_iou((x1, y1, x2, y2), gt)
            ious.append(iou)

    if ious:
        avg_iou = sum(ious) / len(ious)
        #print(f"IoU (dla {filename}): {avg_iou:.2f}")
        all_ious.append(avg_iou)

    time_for_photo = img_stop_time-img_start_time
    times.append(time_for_photo)

    print(f"File:           {filename}")
    print(f"ODCZYTANY:      {plate_text}")
    print(f"Ground Truth:   {gt_plate}")
    print(f"Czas:           {time_for_photo:.2f}")

    results[filename] = (plate_text, gt_plate)

end_time = time.time()
processing_time = end_time - start_time
accuracy = evaluate(results, 1)
fuzzy_accuracy = evaluate(results, 0.9)

print(f"\n📊 Zgodne na 100%: {accuracy:.2f}%")
print(f"📊 Zgodne na  90%: {fuzzy_accuracy:.2f}%")
print(f"⏱️ Czas przetwarzania testowych zdjęć: {processing_time:.2f}s")
time_to_100 = (sum(times) / len(times))*100 if times else 0
print(f"⏱️ Czas szacowany na 100 zdjęć:        {time_to_100:.2f}s")
mean_iou = sum(all_ious) / len(all_ious) if all_ious else 0
print(f"📐 Średnie IoU dla wszystkich obrazów: {mean_iou:.3f}")
grade = calculate_final_grade(accuracy, time_to_100)
print(f"🎓 Ocena końcowa: {grade}")




