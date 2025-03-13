import cv2
import mediapipe as mp
import os
import joblib
import numpy as np
from constants import MP_DRAWING, MP_HANDS, HANDS, CAMERA_INDEX, FRAME_HEIGHT, FRAME_WIDTH, MODEL_PATH

# Configuración de OpenCV
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

model = joblib.load(MODEL_PATH)

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
            MP_DRAWING.draw_landmarks(frame, hand_landmarks, MP_HANDS.HAND_CONNECTIONS)

            landmarks_array = np.array([[int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])] for lm in hand_landmarks.landmark[0:21]])
            if landmarks_array.size > 0:  # Check if landmarks_array is not empty
                try:
                    x, y, w, h = cv2.boundingRect(landmarks_array)
                    hand_roi = cv2.resize(frame[y:y+h, x:x+w], (128, 128), interpolation=cv2.INTER_AREA)
                    hand_roi_gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
                    hand_roi_flatten = hand_roi_gray.flatten()

                    # Realizar la predicción con el modelo SVM
                    predicted_letter = model.predict([hand_roi_flatten])[0]

                    # Mostrar la letra predicha en la ventana
                    cv2.putText(frame, f'Letra: {predicted_letter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                except Exception as e:
                    print(f"Error al procesar la mano: {e}")
    
    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()