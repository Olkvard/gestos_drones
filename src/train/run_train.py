import yaml
import os
from src.train import train_yolov8
import json

def main():
    # 1) Carga de la configuración
    cfg_path = os.path.join("configs", "train_config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # 2) Extraer parámetros
    data        = cfg["data"]
    model_arch  = cfg["model_arch"]
    epochs      = cfg["epochs"]
    batch_size  = cfg["batch_size"]
    img_size    = cfg["img_size"]
    project_dir = cfg["project_dir"]
    exp_name    = cfg["exp_name"]

    # 3) Ejecutar entrenamiento
    print(f"Entrenando YOLOv8 con:\n"
          f"  data       = {data}\n"
          f"  model      = {model_arch}\n"
          f"  epochs     = {epochs}\n"
          f"  batch_size = {batch_size}\n"
          f"  img_size   = {img_size}\n"
          f"  project    = {project_dir}\n"
          f"  exp_name   = {exp_name}\n")
    metrics_path, best_weights = train_yolov8(
        data, model_arch, epochs,
        batch_size, img_size,
        project_dir, exp_name
    )

    # 4) Leer y mostrar métricas finales
    if os.path.isfile(metrics_path):
        with open(metrics_path) as mf:
            metrics = json.load(mf)
        # Mostrar sólo algunas métricas clave
        print("\nMétricas finales (últimos valores):")
        for key, vals in metrics.items():
            last = vals[-1] if isinstance(vals, list) else vals
            print(f"  {key}: {last}")

    print(f"\nMejor peso guardado en: {best_weights}")

if __name__ == "__main__":
    main()