# Detección de gestos con OpenCV y Mediapipe

## Introducción

Este proyecto tiene como objetivo el pilotaje de un dron mediante gestos con las manos, para ello se recogen capturas de los gestos que se utilizarán, se entrena un modelo capaz de reconocer dichos gestos y estos se implementan en el código del dron para su control.

## Tecnologías Utilizadas

- Python
- OpenCV
- MediaPipe
- Scikit-learn (para SVM)
- Numpy

## Requisitos Previos

Antes de comenzar asegurate de instalar las dependencias de `requirements.txt` con:

```bash
pip install requirements.txt
```

## Estructura del Proyecto
```markdown
.
├── data                # Almacena las fotografías tomadas en capture.py
├── capture.py          # Realiza las fotografías para el entrenamiento
├── constants.py        # Contiene las constantes usadas en los archivos
├── train.py            # Entrena el modelo de predicción
├── main.py             # Utiliza el modelo de predicción en tiempo real
├── requirements.txt    # Contiene las dependencias del proyecto
├── .gitignore
└── README.md
```

## Modo de Uso

El primer paso una vez instaladas las dependencias será ajustar los parámetros a los deseados dentro del archivo constants.py. Aquí podremos establecer el número de clases del modelo y los nombres de los gestos, así como los ajustes de la cámara y la configuración de la carpeta de amacenamiento de las imágenes y el modelo.

### Captura de imágenes

Para la captura de las imágenes se debe ejecutar el archivo capture.py. Antes de la ejecución debemos establecer el número de imágenes que se tomarán de cada gesto en la variable capture_count. Este archivo activará la cámara de la propia máquina por defecto y en el momento que detecte una mano comenzará a realizar las capturas y almacenarlas en la carpeta correspondiente, una vez alcanzado el capture_count, se detendrá la captura y tras pulsar una tecla pasará automáticamente a recoger las imágenes para el siguiente gesto.

### Entrenamiento del modelo

Tras la captura de las imágenes podemos ejecutar train.py y el modelo se almacenará automáticamente en la carpeta que hayamos seleccionado.

### Uso de la aplicación

Después de entrenamiento ya podremos ejecutar main.py y realizar las detecciones de los gestos en tiempo real, la predicción del modelo aparecerá arriba a la izquierda en la pantalla de la cámara.

