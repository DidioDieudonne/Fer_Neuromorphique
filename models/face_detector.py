
import cv2
import numpy as np

# Charger le modèle DNN pour la détection des visages
dnn_model_file = "models/res10_300x300_ssd_iter_140000.caffemodel"
dnn_config_file = "models/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(dnn_config_file, dnn_model_file)
print("Modèle DNN chargé avec succès.")

def detect_faces_dnn(image):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x, y, x1, y1 = box.astype("int")
            faces.append((x, y, x1 - x, y1 - y))
    return faces
