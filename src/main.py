import cv2
import joblib
import torch
import numpy as np
from torchvision import models
from PIL import Image
from ultralytics import YOLO
from constants import MP_DRAWING, MP_HANDS, HANDS, CAMERA_INDEX, FRAME_HEIGHT, FRAME_WIDTH, MODEL_PATH, N_CLASSES, SIGNS

# Configuración de OpenCV
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

SKIP = 3
frame_count = 0
last_boxes = []

# Cargar el modelo 
model = YOLO(MODEL_PATH)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No se puede abrir la cámara.")
        break
    
    frame_count += 1

    # Convertir la imagen a RGB para mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar manos
    results = HANDS.process(rgb_frame)

    # Comprobar si se detecta la mano
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            try:
                if frame_count % SKIP == 0:    
                    last_boxes = []
                    results = model(frame, stream=True, conf=0.25, iou=0.45)
                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            # coordenadas absolutas
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            last_boxes.append((x1, y1, x2, y2, cls_id, conf))
                for (x1, y1, x2, y2, cls_id, conf) in last_boxes:
                    label = f"{SIGNS[cls_id]} {conf*100:.2f}%"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)    

            except Exception as e:
                print(f"Error al procesar la mano: {e}")
    
    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()  