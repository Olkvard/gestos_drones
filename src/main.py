import cv2
import joblib
import torch
import numpy as np
from torchvision import models
from PIL import Image
from constants import MP_DRAWING, MP_HANDS, HANDS, CAMERA_INDEX, FRAME_HEIGHT, FRAME_WIDTH, MODEL_PATH, N_CLASSES
from clasificador import preprocess, predict

# Configuración de OpenCV
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if MODEL_PATH == "model/best_model":

    model = models.efficientnet_b0(weights = None, num_classes = N_CLASSES)
    
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    transform = preprocess(USE_AUGMENTATION=False)
else:
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

            # Extraer los puntos de referenci de la mano y calcular el rectángulo que la engloba
            landmarks_array = np.array([[int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])] for lm in hand_landmarks.landmark[0:21]])

            if landmarks_array.size > 0:  # Check if landmarks_array is not empty
                try:
                    x, y, w, h = cv2.boundingRect(landmarks_array)
                    if MODEL_PATH == "model/best_model":
                        # Extraer la región de interés (ROI) de la mano
                        hans_roi = frame[y:y+h, x:x+w]
                        # Redimensionar la ROI a 128x128 (tamaño que espera el modelo)
                        hand_roi = cv2.resize(frame[y:y+h, x:x+w], (128, 128), interpolation=cv2.INTER_AREA)

                        # Convertir la ROI a una imagen PIL en formato RGB
                        pil_image = Image.fromarray(cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY))
                        # Aplicar la transformación predefinida
                        input_tensor = transform(pil_image)
                        input_tensor = input_tensor.unsqueeze(0)
                        input_tensor = input_tensor.to(device)

                        # Realizar la inferencia con el modelo en modo evaluación
                        with torch.no_grad():
                            outputs = model(input_tensor)
                            # Obtener la predicción con mayor probabilidad (top-1)
                            _, predicted = torch.max(outputs.data, 1)
                            predicted_class = predicted.item()
                    else:
                        hand_roi = cv2.resize(frame[y:y+h, x:x+w], (128, 128), interpolation=cv2.INTER_AREA)
                        hand_roi_gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
                        hand_roi_flatten = hand_roi_gray.flatten()
    
                        # Realizar la predicción con el modelo SVM
                        predicted_class = model.predict([hand_roi_flatten])[0]

                    # Mapear la clase predicha a la letra correspondiente

                    # Mostrar la letra predicha en la ventana
                    cv2.putText(frame, f'Letra: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                except Exception as e:
                    print(f"Error al procesar la mano: {e}")
    
    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()