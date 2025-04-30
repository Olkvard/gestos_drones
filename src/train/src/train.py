import os
import glob
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
    return metrics_path, best_weights