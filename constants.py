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

DATA_FOLDER = "data"
MODEL_PATH = "./model/svm_model.pkl"