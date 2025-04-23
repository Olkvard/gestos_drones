# Archivo con las constantes del proyecto

import cv2
import mediapipe as mp

# Inicializamos las Mediapipe Hands
MP_HANDS = mp.solutions.hands
HANDS = MP_HANDS.Hands()
MP_DRAWING = mp.solutions.drawing_utils

# Configuración de la cámara (OpenCV)
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Configuración de la carpeta de datos y el modelo
DATA_FOLDER = "data"
MODEL_PATH = "./model/svm_model.pkl"

# Número de clases para la clasificación
N_CLASSES = 6
SIGNS = ["Strike", "Recce1", "Recce2", "Recce3", "Aterrizar", "Desactivar"]