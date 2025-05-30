# Detector de Gestos para Control de Drones

**Autores:** Alfonso Fuentes, Angel Gutierrez  
**Fecha:** Mayo 2025  

---

## Descripción

Este proyecto implementa un sistema de reconocimiento de gestos manuales para controlar un dron en tiempo real. Usa MediaPipe para detectar y recortar la mano, y luego un modelo de Deep Learning (EfficientNet-B0 finetune) para clasificar 6 gestos básicos:  
- Aterrizar  
- Desactivar  
- Recce1  
- Recce2  
- Recce3  
- Strike  

Además incluye un segundo modelo SVM como baseline y compara resultados en W&B.

---

## Estructura del repositorio
```markdown
.
/
├── src   # Workflows de GitHub Actions
│ └── ci-cd.yml
├── constants.py        # Configuración global (MediaPipe, cámara, rutas)
├── train.ipynb         # Notebook para entrenamiento y validación de modelo
├── main.py             # Inferencia en tiempo real sobre cámara
├── docker-compose.yml  # Definición de servicios (frontend, API, capture, train)
├── Dockerfile.api      # API REST (FastAPI + modelo)
├── Dockerfile.frontend # Frontend web
├── .github/workflows   # Workflows de GitHub Actions
│ └── ci-cd.yml
└── README.md           # Este archivo
```

---

## Requisitos

- Python 3.8+
- Docker & Docker Compose
- Azure CLI
- (Opcional) W&B CLI si quieres reproducir los experiments

---

## Instalación y uso local

1. **Clonar el repo**

   ```bash
   git clone https://github.com/tu-usuario/detector-gestos.git
   cd detector-gestos
   ```
2. **Crear entorno virtual & dependencias**

    ```bash
    python -m venv .venv
    source .venv/bin/activate    # Linux/Mac
    .venv\Scripts\activate       # Windows
    pip install -r requirements.txt
    ```
3. **Capturar imágenes de entrenamiento**

    ```bash
    python capture.py --gesture A --count 100
    ```
4. **Entrenar modelo en Notebook**

    ```bash
    jupyter lab train.ipynb
    ```
    Ajusta hiperparámetros y guarda el modelo en ./model/best_model.pth.

5. **Ejecutar inferencia en tiempo real**

    ```bash
    python main.py
    ```
    Se abrirá una ventana con la cámara y la predicción de gestos.

## Despliegue en producción (Azure App Service)
1. **Pre-requisitos**

    - Cuenta de Azure con App Service Linux (plan Standard+ o Premium).

    - Azure CLI instalada y autenticada (az login).

    - Service Principal con rol Contributor (se guardó JSON en secret AZURE_CREDENTIALS).

2. **Configurar GitHub Secrets**

    - `AZURE_CREDENTIALS`: JSON del SP.

    - `AZURE_RESOURCE_GROUP`: upm_mlops

    - `AZURE_WEBAPP_NAME`: detectorGestos

3. **Pipeline CI/CD**

    Con cada push a `ApiDocker`, GitHub Actions:

    - Se autentica (azure/login@v1).

    - Construye frontend (si aplica).

    - Ejecuta

    ```bash
    az webapp config container set \
      --resource-group ${{ secrets.AZURE_RESOURCE_GROUP }} \
      --name ${{ secrets.AZURE_WEBAPP_NAME }} \
      --multicontainer-config-type compose \
      --multicontainer-config-file docker-compose.yml
    az webapp restart … 
    ```
    - Despliega los 2 contenedores (frontend, API) según docker-compose.prod.yml.

4. **Acceso al servicio**

    - Frontend: `https://detectorgestos-a6b4ghg2fpddfwfz.spaincentral-01.azurewebsites.net`


## Enlaces
- GitHub: https://github.com/Olkvard/gestos_drones/tree/ApiDocker

- W&B Dashboard: https://wandb.ai/a-fuentesr-universidad-politecnica-de-madrid/Deteccion_de_gestos?nw=nwuserafuentesr

- Endpoint Azure: https://detectorgestos-a6b4ghg2fpddfwfz.spaincentral-01.azurewebsites.net