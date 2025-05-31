import cv2
import os
import numpy as np
import json
import mediapipe as mp

# Configuración de OpenCV
capture_count = 0
i = 0

MP_HANDS = mp.solutions.hands
HANDS = MP_HANDS.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.7,
)

DATA_FOLDER = os.environ.get("DATA_PATH", "/app/images")

CAMERA_INDEX = os.environ.get("CAMERA_INDEX", 0)
FRAME_WIDTH = os.environ.get("FRAME_WIDTH", 640)
FRAME_HEIGHT = os.environ.get("FRAME_HEIGHT", 480)
SIGNS = json.loads(os.getenv("SIGNS", "[]"))

cap = cv2.VideoCapture(int(CAMERA_INDEX))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(FRAME_WIDTH))
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(FRAME_HEIGHT))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No se puede abrir la cámara.")
        break

    # Convertir la imagen a RGB para mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar manos
    results = HANDS.process(rgb_frame)

    # Comprobar si se detecta la mano
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dibuja los puntos de la mano en la imagen

            # Captura una imagen de la mano cada frame
            # (Elimina la condición de cada 30 frames para capturar más rápido)
            # Además, verifica si hemos capturado suficientes imágenes para la letra actual
            if capture_count < 500:

                hand_roi = frame.copy()  # Copia la imagen original
                # Guarda la imagen en la carpeta correspondiente a la letra del abecedario
                folder_path = f"{DATA_FOLDER}/{SIGNS[i]}"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                cv2.imwrite(f"{folder_path}/{SIGNS[i]}_{capture_count}.png", hand_roi)

                print(f"Imagen {capture_count} capturada para el gesto {SIGNS[i]}")
                capture_count += 1
            else:
                # Pide presionar una tecla para continuar con la siguiente letra
                cv2.putText(frame, f'Presiona una tecla para continuar con el gesto {SIGNS[i]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow("Hand Tracking", frame)
                cv2.waitKey(0)  # Espera hasta que se presione una tecla
                # Reinicia el contador y pasa a la siguiente letra
                capture_count = 0
                i += 1

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()