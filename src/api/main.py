from dotenv import load_dotenv
import os
import cv2
from ultralytics import YOLO
from constants import HANDS
import json
import wandb

load_dotenv()

CAMERA_INDEX = os.environ.get("CAMERA_INDEX", "0")
FRAME_WIDTH = os.environ.get("FRAME_WIDTH", "640")
FRAME_HEIGHT = os.environ.get("FRAME_HEIGHT", "480")
SIGNS = json.loads(os.getenv("SIGNS", "[]"))
# Configuración de OpenCV
cap = cv2.VideoCapture(int(CAMERA_INDEX))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(FRAME_WIDTH))
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(FRAME_HEIGHT))
print("Cámara abierta")

SKIP = 3
frame_count = 0
last_boxes = []

wandb.init(
    project="Deteccion_de_gestos",
    name="HandTracking",
    config={
        "camera_index": CAMERA_INDEX,
        "frame_width": FRAME_WIDTH,
        "frame_height": FRAME_HEIGHT,
        "skip_frames": SKIP,
        "signs": SIGNS
    }
)

# Cargar el modelo de detección de manos
artifact_path = os.environ.get("WANDB_MODEL", "a-fuentesr-universidad-politecnica-de-madrid/Deteccion_de_gestos/svm_model:latest")
artifact = wandb.use_artifact(artifact_path, type="model")

# Cargar el modelo 
model = YOLO(artifact)
print("Modelo cargado")

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
                    label = f"{SIGNS[cls_id]} {conf*100:.2f}%" if cls_id < len(SIGNS) else "Unknown"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)    

            except Exception as e:
                print(f"Error al procesar la mano: {e}")
    
    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()  