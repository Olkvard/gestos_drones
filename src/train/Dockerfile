# Dockerfile
FROM python:3.8-slim

RUN apt-get update && apt-get install -y curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the entire api folder to the container
COPY . .

# By default, let's run the main script
CMD ["python", "run_train.py"]