import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import wandb

DATA_FOLDER = "images"
SIGNS = ["1Dedo", "2Dedos", "3Dedos", "Mano Abierta", "Manos Arriba", "Puño"]

current_version = 1.9
C_value = 1
kernel_type = 'poly'  # Cambia a 'precomputed', 'sigmoid', 'rbf', 'poly', 'linear'

# Crear una nueva ejecución de wandb para registrar este archivo
run = wandb.init(
    name="Entrenamiento SVM",
    # Configurar el nombre del proyecto
    project="Deteccion_de_gestos",
    # Registrar los hiperparámetros y metadata
    config={
        "C": C_value,
        "kernel": kernel_type,
        "version": current_version,
    },
)

# Cargar imágenes de entrenamiento
X, y = [], []

for sign in SIGNS:
    for i in range(500):
        img_path = f"{DATA_FOLDER}/{sign}/{sign}_{i}.png"
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (128, 128))  # Ajusta el tamaño según sea necesario
            X.append(img.flatten())
            y.append(sign)

X = np.array(X)
y = np.array(y)

print("Dataset cargado:", X.shape, y.shape)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------------------------------------------------
# 1. Función de entrenamiento del modelo SVM (ejecutada fuera del bloque MLflow)
# -------------------------------------------------------------------------
def train_svm(X_train, y_train, X_val, y_val, C=1.0, kernel='linear'):
    # Crear el modelo SVM con el kernel y parámetro C indicados. Se utiliza
    # probability=True para poder usar la función predict_proba si se desea.
    model = SVC(C=C, kernel=kernel, probability=True)
    
    # Entrenamiento
    model.fit(X_train, y_train)
    
    # Predicción en el conjunto de validación
    y_pred = model.predict(X_val)
    # Calcular la precisión
    accuracy = accuracy_score(y_val, y_pred)
    # Generar un reporte de clasificación para ver más detalles
    report = classification_report(y_val, y_pred)
    
    return model, accuracy, report


# -------------------------------------------------------------------------
# 3. Entrenar el modelo SVM
# -------------------------------------------------------------------------

svm_model, val_accuracy, val_report = train_svm(X_train, y_train, X_test, y_test, C=C_value, kernel=kernel_type)

print("Reporte de clasificación en Validación:\n", val_report)
print("Precisión en Validación:", val_accuracy)

artifact = wandb.Artifact("svm_model", type="model")

joblib.dump(svm_model, "artifacts/svm_model.pkl")

# Guardar el modelo en el artefacto
artifact.add_file("artifacts/svm_model.pkl")

# Registrar el artefacto en wandb

# Registra las métricas en wandb
run.log({"acc": val_accuracy})

run.finish()
