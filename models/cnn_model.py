
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import models, transforms
from models.face_detector import detect_faces_dnn
from models.utils import extract_features, labels
import torch.nn.functional as F

def load_cnn_model(weights_path="models/cnn_model_p.pth"):
    model = models.resnet50(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, len(labels))
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    print("Modèle CNN (ResNet50) chargé avec succès.")
    return model


# Prédiction d'émotion avec le modèle CNN


def predict_emotion_cnn(image, cnn_model):
    faces = detect_faces_dnn(image)
    results = []
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    for (x, y, w, h) in faces:
        face_gray = cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (48, 48))
        img_tensor = transform(face_resized).unsqueeze(0)

        with torch.no_grad():
            pred = cnn_model(img_tensor)
            probabilities = F.softmax(pred, dim=1)  # Appliquez softmax pour obtenir des probabilités
            confidence, pred_class = torch.max(probabilities, 1)  # Confidence sera compris entre 0 et 1
            if confidence.item() > 0.4:
                results.append({
                    'position': (x, y, w, h),
                    'emotion': labels[pred_class.item()],
                    'confidence': confidence.item() * 1  # Multipliez par 100 pour avoir des pourcentages
                })
    return results





