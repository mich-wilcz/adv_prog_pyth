import os
import shutil
import subprocess
import easyocr

# Ścieżki
SOURCE_IMAGES_DIR = 'cuts'
LABEL_FILE = 'plates_labels.txt'
TRAIN_DIR = 'train_ocr'
IMAGES_DIR = os.path.join(TRAIN_DIR, 'images')
TRAIN_LABELS_PATH = os.path.join(TRAIN_DIR, 'labels.txt')
MODEL_OUTPUT_DIR = 'my_ocr_model'  # miejsce zapisu modelu
TEST_IMAGE_PATH = os.path.join('test.jpg')  # przykładowy obrazek do testu

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

print("Kopiowanie obrazów...")
for filename in os.listdir(SOURCE_IMAGES_DIR):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        shutil.copy(os.path.join(SOURCE_IMAGES_DIR, filename), os.path.join(IMAGES_DIR, filename))

print("Generowanie labels.txt...")
with open(LABEL_FILE, 'r', encoding='utf-8') as fin, open(TRAIN_LABELS_PATH, 'w', encoding='utf-8') as fout:
    for line in fin:
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        filename, label = parts
        fout.write(f'images/{filename} {label}\n')

TRAIN_CMD = [
    'python', 'EasyOCR/trainer/train.py',
    '--train_data', TRAIN_DIR,
    '--val_data', TRAIN_DIR,
    '--lang_list', "['en']",
    '--num_epochs', '30',
    '--experiment_name', 'my_ocr'
]

print("Rozpoczynanie treningu EasyOCR...")
ret = subprocess.run(TRAIN_CMD)

if ret.returncode != 0:
    print("Trening zakończył się błędem.")
    exit(1)

print("Trening zakończony pomyślnie.")

experiment_dir = os.path.join('output', 'rec_multi_language_lite')

if not os.path.exists(experiment_dir):
    print(f"Folder eksperymentu '{experiment_dir}' nie istnieje.")
    exit(1)

if os.path.exists(MODEL_OUTPUT_DIR):
    shutil.rmtree(MODEL_OUTPUT_DIR)

shutil.copytree(experiment_dir, MODEL_OUTPUT_DIR)
print(f"Model skopiowany do folderu '{MODEL_OUTPUT_DIR}'.")

print("Testowanie modelu na obrazie testowym...")
reader = easyocr.Reader(['en'], model_storage_directory=MODEL_OUTPUT_DIR, user_network_directory=MODEL_OUTPUT_DIR, gpu=True)
result = reader.readtext(TEST_IMAGE_PATH)

print("Wyniki dla testu OCR:")
for bbox, text, prob in result:
    print(f"Tekst: {text} | Pewność: {prob:.2f}")
