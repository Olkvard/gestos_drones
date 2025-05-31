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
├── .github/workflows
│ └── ci-cd.yml
├── src
│ └── api 
│   └── Dockerfile
│   └── constants.py
│   └── main.py
│   └── requirements.txt
│ └── capture
│   └── Dockerfile
│   └── YOLOdataset.py
│   └── capture.py
│   └── requirements.txt
│ └── frontend
│   └── templates
│       └── index.html
│   └── Dockerfile
│   └── app.py
│   └── requirements.txt
│ └── train
│   └── configs
│       └── train_config.yaml
│   └── src
│       └── train.py
│   └── Dockerfile
│   └── run_train.py
│   └── train_svm.py
│   └── requirements.txt
│ └── docker-compose.prod.yml
│ └── docker-compose.yml
└── README.md
```

---

## Requisitos

- Python 3.8+
- Docker & Docker Compose
- Azure CLI
- W&B CLI si quieres reproducir los experiments

---

## Instalación y uso local

1. **Clonar el repo**

   ```bash
   git clone https://github.com/tu-usuario/detector-gestos.git
   cd detector-gestos
   ```
2. **Creación del .env**
    
    Será necesaria la creación de un archivo .env con las variables de entorno para el proyecto, este archivo deberá incluir la clave de wandb, el resto de de variables ya cuentan con un valor por defecto, en caso de querer cambiarlas deberán incluirse en el archivo.

3. **Ejecutar las imágenes de Docker**

    Ejecutar la imágen de captura de imágenes de Docker, esto abrirá una ventana con OpenCV y permitirá realizar la captura de las imágenes para el entrenamiento. La captura realiza 500 fotografías para cada gesto.

4. **Entrenar modelo en la imágen train**

    Ejecutar la imágen train, la cuál realizará el entrenamiento del modelo y lo subirá a wandb donde se podrán ver los resultados del entrenamiento.

5. **Ejecutar imágenes api y web**

    Esto abrirá una página web hosteada en un puerto de la máquina donde se podrá subir una imágen y el modelo la analizará para comprobar el gesto que se encuentra en dicha imágen.

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

- W&B Dashboard: [https://wandb.ai/a-fuentesr-universidad-politecnica-de-madrid/Deteccion_de_gestos?nw=nwuserafuentesr](https://api.wandb.ai/links/a-fuentesr-universidad-politecnica-de-madrid/ewwd1b6w)

- Endpoint Azure: https://detectorgestos-a6b4ghg2fpddfwfz.spaincentral-01.azurewebsites.net
