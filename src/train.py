import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import matplotlib.pyplot as plt
import mlflow
from mlflow.models import infer_signature
from constants import DATA_FOLDER, MODEL_PATH, SIGNS

current_version = 1.5
mlflow.set_tracking_uri("http://localhost:5000")

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
    clf = SVC(C=C, kernel=kernel, probability=True)
    
    # Entrenamiento
    clf.fit(X_train, y_train)
    
    # Predicción en el conjunto de validación
    y_pred = clf.predict(X_val)
    # Calcular la precisión
    accuracy = accuracy_score(y_val, y_pred)
    # Generar un reporte de clasificación para ver más detalles
    report = classification_report(y_val, y_pred)
    
    return clf, accuracy, report


# -------------------------------------------------------------------------
# 3. Entrenar el modelo SVM
# -------------------------------------------------------------------------
C_value = 0.1
kernel_type = 'rbf'  # Cambia a 'precomputed', 'sigmoid', 'rbf', 'poly', 'linear'

svm_model, val_accuracy, val_report = train_svm(X_train, y_train, X_test, y_test, C=C_value, kernel=kernel_type)

print("Reporte de clasificación en Validación:\n", val_report)
print("Precisión en Validación:", val_accuracy)

# Además, guarda localmente el modelo (opcional)
joblib.dump(svm_model, MODEL_PATH)

# -------------------------------------------------------------------------
# 4. Registrar en MLflow
# Ejecuta la parte de logging en el bloque mlflow.start_run(), pero no engloba la
# ejecución del entrenamiento.
# -------------------------------------------------------------------------
# Configura el experimento (se crea o se selecciona uno existente).
mlflow.set_experiment("SVM_Gestures_Experiment")

with mlflow.start_run(run_name=f"Modelo SVM {current_version}") as run:
    # Registrar parámetros
    mlflow.log_param("C", C_value)
    mlflow.log_param("kernel", kernel_type)
    
    # Registrar métricas
    mlflow.log_metric("val_accuracy", val_accuracy)

    # Poner un tag para identificar el modelo
    mlflow.set_tag("model_version", current_version)
    
    # Registrar el reporte de clasificación (se guarda en un archivo de texto)
    report_file = "classification_report.txt"
    with open(report_file, "w") as f:
        f.write(val_report)
    mlflow.log_artifact(report_file, artifact_path="reports")
    
    # Registrar el modelo SVM con mlflow.sklearn
    mlflow.sklearn.log_model(svm_model, "svm_model")
    
    print("Run ID:", run.info.run_id)