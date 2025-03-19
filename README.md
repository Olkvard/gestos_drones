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

