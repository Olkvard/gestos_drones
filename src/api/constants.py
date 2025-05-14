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