import cv2
import mediapipe as mp
import os
import numpy as np

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Configuración de OpenCV
cap = cv2.VideoCapture(0)  # Usa 0 para la cámara predeterminada, puedes cambiar esto según tus necesidades
capture_count = 0
letter_count = 0
current_letter = 'a'  # Inicializar con la primera letra

# Nuevo import para drawing_utils
mp_drawing = mp.solutions.drawing_utils

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No se puede abrir la cámara.")
        break

    # Convertir la imagen a RGB para mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar manos
    results = hands.process(rgb_frame)

    # Comprobar si se detecta la mano
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dibuja los puntos de la mano en la imagen
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()