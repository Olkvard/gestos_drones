from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from constants import HANDS
import cv2
import os
import json
import wandb
import joblib
import uvicorn

API_PORT = int(os.environ.get("API_PORT", 8000))

SIGNS = json.loads(os.getenv("SIGNS", '["1Dedo", "2Dedos", "3Dedos", "Mano Abierta", "Manos Arriba", "Puño"]'))

api = wandb.Api()

app = FastAPI()

# Cargar el modelo al iniciar la aplicación
artifact_path = os.environ.get("WANDB_MODEL", "a-fuentesr-universidad-politecnica-de-madrid/Deteccion_de_gestos/svm_model:v0")
print(f"Artifact path: {artifact_path}")
artifact = api.artifact(artifact_path, type="model")
artifact_dir = artifact.download()

# Cargar el modelo SVM desde el archivo descargado
model_path = os.path.join(artifact_dir, "svm_model.pkl")  # Asegúrate de que el archivo tenga este nombre
if not os.path.exists(model_path):
    raise FileNotFoundError(f"No se encontró el archivo del modelo SVM en {model_path}")
model = joblib.load(model_path)
print("Modelo SVM cargado correctamente.")

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Guardar la imagen temporalmente
        image_path = f"/tmp/{file.filename}"
        with open(image_path, "wb") as buffer:
            buffer.write(await file.read())

        # Leer la imagen
        frame = cv2.imread(image_path)
        if frame is None:
            raise HTTPException(status_code=400, detail="No se pudo cargar la imagen")

        # Preprocesar la imagen (por ejemplo, convertirla a escala de grises y redimensionarla)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, (64, 64))  # Ajusta el tamaño según lo que espera tu modelo
        features = resized_frame.flatten().reshape(1, -1)  # Convertir la imagen en un vector de características

        # Realizar la predicción con el modelo SVM
        prediction = model.predict(features)[0]

        # Devolver el resultado como JSON
        gesture = SIGNS[prediction] if prediction < len(SIGNS) else "Unknown"
        
        return JSONResponse(content={"predictions": gesture})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=API_PORT)