# Archivo con las constantes del proyecto

import cv2
import mediapipe as mp

# Inicializamos las Mediapipe Hands
MP_HANDS = mp.solutions.hands
HANDS = MP_HANDS.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.7,
)
MP_DRAWING = mp.solutions.drawing_utils

# Configuración de la cámara (OpenCV)
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Configuración de la carpeta de datos y el modelo
DATA_FOLDER = "images"
MODEL_PATH = "./runs/gesture_exp/weights/best.pt"

# Número de clases para la clasificación
N_CLASSES = 6
SIGNS = ["Puño", "1Dedo", "2Dedos", "3Dedos", "Mano Abierta", "Manos Arriba"]

# Parámetros de entrenamiento YOLO

DATA_YAML = "/home/alfonso.fuentes@sener.es/Documentos/proyectos/drones_control/yolo_dataset/data.yaml"
MODEL_ARCH = "yolov8s.pt"
EPOCHS = 50
BATCH_SIZE = 16
IMG_SIZE = 640
PROJECT_DIR = "runs"