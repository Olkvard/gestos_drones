FROM python:3.8-slim

# Instalar dependencias del sistema (y v4l2 si quieres hacer capture)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia e instala pip deps
COPY requirements.txt .
RUN pip install --default-timeout=100 -r requirements.txt

# Copia el código fuente
COPY . .

CMD ["python", "run_train.py"]
