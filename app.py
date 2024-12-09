

from flask import Flask, render_template, request, redirect
import base64
import cv2
import numpy as np
from models.cnn_model import load_cnn_model, predict_emotion_cnn
from models.snn_predictor import predict_emotion_snn
from models.snn_model import load_snn_model


app = Flask(__name__)

# Charger les modèles au démarrage
cnn_model = load_cnn_model()
snn_model = load_snn_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    npimg = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Prédiction avec les modèles CNN et SNN
    cnn_results = predict_emotion_cnn(image, cnn_model)
    snn_results = predict_emotion_snn(image)

    # Annoter l'image avec les résultats CNN et SNN
    for result in cnn_results:
        x, y, w, h = result['position']
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, result['emotion'], (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1)

    for result in snn_results:
        x, y, w, h = result['position']
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, result['emotion'], (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)

    _, img_encoded = cv2.imencode('.png', image)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    return render_template('results.html', cnn_results=cnn_results, snn_results=snn_results, image_data=img_base64)

@app.route('/thank_you')
def thank_you():
    return render_template('thank_you.html')


if __name__ == '__main__':
    app.run(debug=True)
