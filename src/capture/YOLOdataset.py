import os
import glob
import random
import shutil
import numpy as np
import wandb
import logging
import yaml
import json
from pathlib import Path
from ultralytics.data.dataset import DATASET_CACHE_VERSION
from ultralytics.data.utils import save_dataset_cache_file

import cv2
import mediapipe as mp

# Inicializa wandb
run = wandb.init(project="Deteccion_de_gestos", name="Generar dataset YOLOv8")

# Crear artifact
artifact = wandb.Artifact("gestures_dataset", type="dataset", description="Dataset preparado para YOLOv8")

DATA_FOLDER =  os.getenv("DATA_PATH", "images")
if not os.path.exists(DATA_FOLDER):
    raise FileNotFoundError(f"DATA_FOLDER '{DATA_FOLDER}' does not exist.")
MP_HANDS = mp.solutions.hands
HANDS = MP_HANDS.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.7,
)

OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "yolo_dataset")
SPLITS = {"train": 0.7, "validation": 0.2, "test": 0.1}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
for split in SPLITS.keys():
    os.makedirs(os.path.join(OUTPUT_FOLDER, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, split, "labels"), exist_ok=True)

gesture_folders = sorted([
    d for d in os.listdir(DATA_FOLDER)
    if os.path.isdir(os.path.join(DATA_FOLDER, d))
])

class_index = {gesture: i for i, gesture in enumerate(gesture_folders)}
print("Índice de clases:", class_index)

# Recopila todas las imágenes y sus etiquetas
all_images = []
for gesture in gesture_folders:
    for img_path in glob.glob(os.path.join(DATA_FOLDER, gesture, "*.png")):
        all_images.append((img_path, class_index[gesture]))
        
# Baraja y divide según los porcentajes de SPLITS
random.shuffle(all_images)
n = len(all_images)
start = 0
splitted_images = {}
for split, frac in SPLITS.items():
    cnt = int(n * frac)
    splitted_images[split] = all_images[start:start + cnt]
    start += cnt
# Ajusta cualquier sobrante al último split
if start < n:
    splitted_images["test"].extend(all_images[start:])
    
# Crea las carpetas de salida
for split in SPLITS:
    os.makedirs(os.path.join(OUTPUT_FOLDER, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, split, "labels"), exist_ok=True)
    
two_hand_id = class_index.get("Manos Arriba", None)
if two_hand_id is None:
    print("No se encontró la clase 'Manos Arriba' en el índice de clases.")
    
# Función para procesar una imagen: detectar mano y devolver bbox normalizado
def get_hand_bbox_normalized(img_path, class_id):
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = HANDS.process(rgb)
    if not results.multi_hand_landmarks:
        return None
    all_boxes = []
    for hand_landmarks in results.multi_hand_landmarks:
        pts = np.array([
            [int(lm.x * w), int(lm.y * h)]
            for i, lm in enumerate(hand_landmarks.landmark)
            if i != 0  # descartamos el landmark 0
        ])
        x,y,ww,hh = cv2.boundingRect(pts)
        all_boxes.append((x,y,ww,hh))
        
    if class_id == two_hand_id and len(all_boxes) >= 2:
        xs = [b[0] for b in all_boxes] + [b[0] + b[2] for b in all_boxes]
        ys = [b[1] for b in all_boxes] + [b[1] + b[3] for b in all_boxes]
        x, y = min(xs), min(ys)
        ww, hh = max(xs) - x, max(ys) - y
    else:
        x, y, ww, hh = max(all_boxes, key=lambda b: b[2] * b[3])
    
    # Normaliza la bbox
    x_c = (x + ww / 2) / w
    y_c = (y + hh / 2) / h
    return x_c, y_c, ww / w, hh / h
    
# Recorre cada split y genera las imágenes y etiquetas
for split, items in splitted_images.items():
    labels_cache = {
        "images": 0,
        "annotations": 0,
        "classes": set(),
    }

    # Crea las carpetas de destino si no existen
    imgs_dir = os.path.join(OUTPUT_FOLDER, split, "images")
    lbls_dir = os.path.join(OUTPUT_FOLDER, split, "labels")
    os.makedirs(imgs_dir, exist_ok=True)
    os.makedirs(lbls_dir, exist_ok=True)
    for img_path, cls in items:
        fn = os.path.basename(img_path)
        name, ext = os.path.splitext(fn)
        
        # Copia la imagen
        dst_img_path = os.path.join(OUTPUT_FOLDER, split, "images", fn)
        shutil.copy(img_path, dst_img_path)
        
        # Calcula la bbox normalizada
        bbox = get_hand_bbox_normalized(img_path, cls)
        if bbox is None:
            continue
        else:
            x_min, y_min, x_max, y_max = bbox
            
        # Escribe el label.txt
        label_path = os.path.join(OUTPUT_FOLDER, split, "labels", f"{name}.txt")
        with open(label_path, "w") as f:
            f.write(f"{cls} {x_min:.6f} {y_min:.6f} {x_max:.6f} {y_max:.6f}\n")
        
        labels_cache["images"] += 1
        labels_cache["annotations"] += 1
        labels_cache["classes"].add(cls)
        
    labels_cache["classes"] = list(labels_cache["classes"])
    labels_cache["version"] = DATASET_CACHE_VERSION
    cache_path = Path(OUTPUT_FOLDER) / split / "labels.cache"
    save_dataset_cache_file(
        prefix="",
        path=cache_path,
        x=labels_cache,
        version=DATASET_CACHE_VERSION
    )
            
def create_data_yaml(output_folder, class_index):
    """
    Crea el archivo data.yaml para YOLOv8._
    """
    data_yaml_path = os.path.join(output_folder, "data.yaml")
    yaml_content = {
        "train": "train/images",
        "val": "validation/images",
        "test": "test/images",
        "nc": len(class_index),
        "names": list(class_index.keys())
    }
    
    with open(data_yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
        
    return data_yaml_path

data_yaml_path = create_data_yaml(OUTPUT_FOLDER, class_index)

logging.basicConfig(level=logging.INFO)
logging.info("Dataset YOLO generated in: %s", OUTPUT_FOLDER)

artifact.add_dir(OUTPUT_FOLDER)
artifact.add_file(data_yaml_path)

run.log_artifact(artifact)
run.finish()
            
            