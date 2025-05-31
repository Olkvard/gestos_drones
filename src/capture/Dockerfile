# Dockerfile modificado
FROM python:3.8-slim

# Dependencias de sistema (incluyendo V4L2)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libv4l-0 \
    libv4l-dev \
    v4l-utils \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements e instalar con timeout ampliado
COPY requirements.txt .
# Opción 1: usar --default-timeout
RUN pip install --default-timeout=100 -r requirements.txt

# Copiar el código fuente
COPY . .

# Forzar preload de V4L2
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libv4l/v4l2convert.so

CMD ["python", "capture.py"]