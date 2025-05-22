# Dockerfile
# Imagen base
FROM python:3.8-slim

# Variables de entorno
RUN apt-get update && apt-get install -y curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar las dependencias e instalarlas
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copiar el resto del código fuente
COPY . .

# Por default, el contenedor ejecutará el script capture.py
CMD ["python", "YOLOdataset.py"]