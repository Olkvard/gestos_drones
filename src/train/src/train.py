import os
import glob
import wandb
from ultralytics import YOLO

def train_yolov8(data: str, model_arch: str, epochs: int,
                 batch_size: int, img_size: int,
                 project_dir: str, exp_name: str):
    """
    Entrena YOLOv8 con los parámetros dados usando rutas absolutas.
    Devuelve la ruta a metrics.json y al best.pt final.
    """
    # Asegúrate de que 'data' es absoluta
    data = os.path.abspath(data)
    
    wandb.init(
        project="Deteccion_de_gestos",
        name=exp_name,
        config={
            "data": data,
            "model_arch": model_arch,
            "epochs": epochs,
            "batch_size": batch_size,
            "img_size": img_size,
        }
    )
    
    data = os.getenv("YOLO_DATASET", "/app/yolo_dataset/data.yaml")
    if not os.path.isfile(data):
        raise FileNotFoundError(f"El archivo de datos '{data}' no existe. Asegúrate de que la ruta es correcta.")

    # Carga YOLOv8-small
    model = YOLO(model_arch)

    # Eliminamos callbacks internos (para no invocar MLflow u otros)
    model.callbacks = []

    # Lanzamos el entrenamiento
    model.train(
        data     = data,
        epochs   = epochs,
        batch    = batch_size,
        imgsz    = img_size,
        project  = project_dir,
        name     = exp_name,
        exist_ok = True
    )

    # Carpeta donde YOLOv8 guarda resultados
    run_dir      = os.path.join(project_dir, exp_name)
    metrics_path = os.path.join(run_dir, "metrics.json")
    weights_dir  = os.path.join(run_dir, "weights")
    # Obtener el fichero best.pt
    best_weights = glob.glob(os.path.join(weights_dir, "best*.pt"))[0]
    
    # Guardar el artefacto del modelo en wandb
    model_artifact = wandb.Artifact(
        name="gestures_model",
        type="model",
        description="Modelo YOLOv8 entrenado para detección de gestos",
        metadata={
            "metrics_path": metrics_path,
            "epochs": epochs,
            "batch_size": batch_size,
            "img_size": img_size,
            "model_arch": model_arch,
        }
    )
    model_artifact.add_file(best_weights)
    wandb.log_artifact(model_artifact)
    
    # Cerrar wandb
    wandb.finish()
    return metrics_path, best_weights