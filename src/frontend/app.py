from flask import Flask, request, jsonify, render_template
import requests
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

API_HOST = os.environ.get("API_HOST", "api")
API_PORT = os.environ.get("API_PORT", "8001")
API_URL = f"http://{API_HOST}:{API_PORT}/analyze"

# Puerto de la aplicación web
WEB_APP_PORT = os.environ.get("WEB_APP_PORT", "8080")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    image.save(image_path)

    try:
        with open(image_path, 'rb') as img_file:
            response = requests.post(API_URL, files={'image': img_file})
            return jsonify(response.json())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=WEB_APP_PORT, debug=True)