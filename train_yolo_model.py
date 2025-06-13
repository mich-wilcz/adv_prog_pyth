#!pip install easyocr
# !apt install tesseract-ocr -y
# !pip install opencv-python-headless
# !pip install pytesseract
# !pip install ultralytics
# !pip install paddleocr

# --- IMPORTY ---

import os
import cv2
import numpy as np
from ultralytics import YOLO
import re
import time
import xml.etree.ElementTree as ET
import shutil
from sklearn.model_selection import train_test_split
import yaml
import subprocess

ANNOTATIONS_FILE = 'annotations.xml'  # Plik XML z adnotacjami
IMAGE_DIR = 'photos'  # Folder z obrazami
YOLO_DIR = 'yolo_dataset'  # Folder docelowy dla YOLO


def convert_to_yolo_bbox(img_width, img_height, xtl, ytl, xbr, ybr):
    x_center = ((xtl + xbr) / 2) / img_width
    y_center = ((ytl + ybr) / 2) / img_height
    width = (xbr - xtl) / img_width
    height = (ybr - ytl) / img_height
    return x_center, y_center, width, height

def parse_annotations(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    samples = []
    for image in root.findall('image'):
        filename = image.get('name')
        width = int(image.get('width'))
        height = int(image.get('height'))
        for box in image.findall('box'):
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            bbox = convert_to_yolo_bbox(width, height, xtl, ytl, xbr, ybr)
            samples.append((filename, bbox))
    return samples

samples = parse_annotations(ANNOTATIONS_FILE)
unique_files = list(set([s[0] for s in samples]))
train_files, test_files = train_test_split(unique_files, test_size=0.3, random_state=42)

with open("test_files.txt", "w") as f:
    for file in test_files:
        f.write(file + "\n")

for subdir in ['train/images', 'train/labels', 'val/images', 'val/labels']:
    os.makedirs(os.path.join(YOLO_DIR, subdir), exist_ok=True)

for split, files in [('train', train_files), ('val', test_files)]:
    for filename in files:
        # kopiowanie zdjęcia
        src_img_path = os.path.join(IMAGE_DIR, filename)
        dst_img_path = os.path.join(YOLO_DIR, f'{split}/images', filename)
        shutil.copy(src_img_path, dst_img_path)

        # tworzenie etykiet
        labels = [bbox for fname, bbox in samples if fname == filename]
        label_path = os.path.join(YOLO_DIR, f'{split}/labels', filename.replace('.jpg', '.txt'))
        with open(label_path, 'w') as f:
            for bbox in labels:
                x, y, w, h = bbox
                f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")  # class 0 = plate

yaml_content = f'''
path: {YOLO_DIR}
train: train/images
val: val/images

nc: 1
names: ['plate']
'''

with open(os.path.join(YOLO_DIR, 'data.yaml'), 'w') as f:
    f.write(yaml_content.strip())

# TRENING YOLOv8
model = YOLO('yolov8n.pt')
model.train(
    data=os.path.join(YOLO_DIR, 'data.yaml'),
    epochs=50,
    imgsz=640,
    batch=16,
    name='plate_yolov8'
)


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


def crop_plate_strip(img_to_crop, margin_ratio=0.07):
    h, w = img_to_crop.shape[:2]
    x_margin = int(w * margin_ratio)
    cropped = img_to_crop[:, x_margin:]  # ucinamy z lewej strony
    return cropped


def przytnij_marginesy(img, margines_x=30, margines_y=18):
    h, w = img.shape[:2]
    return img[margines_y:h - margines_y, margines_x:w - margines_x]


model = YOLO('runs/detect/plate_yolov8/weights/best.pt')
samples = parse_annotations('annotations.xml')
unique_files = list(set([s[0] for s in samples]))

output_folder = 'cuts'
os.makedirs(output_folder, exist_ok=True)
annotations = load_annotations('annotations.xml')
annotations_dict = {name: label for name, _, label in annotations}

labels_file = open('plates_labels.txt', 'w', encoding='utf-8')

for i, filename in enumerate(train_files):
    path = os.path.join("photos", filename)
    image = cv2.imread(path)
    if image is None:
        continue

    boxes = [box for name, box, _ in annotations if name == filename]
    if not boxes:
        continue
    gt_plate = annotations_dict.get(filename, "BRAK")

    # YOLO + deskew + OCR
    results = model(image)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate_crop = image[y1:y2, x1:x2]

        aligned_plate = crop_plate_strip(plate_crop)

        plate_crop = image[y1:y2, x1:x2]

        step = prostuj_tablice(plate_crop)

        # Przeskaluj obraz do 800x200 (szerokość x wysokość)
        img_resized = cv2.resize(step, (800, 200))

        img_resized = przytnij_marginesy(img_resized)

        # Zbuduj nazwę pliku, np. <oryginalna_nazwa>_plate_<nr>.jpg
        base_name = os.path.splitext(filename)[0]
        save_name = f"{base_name}.jpg"
        save_path = os.path.join(output_folder, save_name)

        cv2.imwrite(save_path, img_resized)

        # Zapis do pliku: nazwa pliku + tab + etykieta + nowa linia
        labels_file.write(f"{base_name}.jpg\t{gt_plate}\n")

labels_file.close()

