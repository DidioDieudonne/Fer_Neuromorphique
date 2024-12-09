
import torch
import torch.nn.functional as F
import cv2
from snntorch import spikegen
from models.snn_model import load_snn_model
from models.face_detector import detect_faces_dnn
from models.utils import extract_features, labels
from torchvision import transforms

# Charger le modèle SNN
snn_model = load_snn_model()

# Paramètres pour l'encodage
num_steps = 20
batch_size = 32

# Définir la transformation de normalisation
normalize = transforms.Normalize((0.5,), (0.5,))

import torch.nn.functional as F

def predict_emotion_snn(image):
    faces = detect_faces_dnn(image)
    results = []

    for (x, y, w, h) in faces:
        face_gray = cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (48, 48))
        img = extract_features(face_resized)

        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        img_tensor = normalize(img_tensor)

        spike_data = spikegen.rate(img_tensor, num_steps=num_steps)
        spike_data = spike_data.repeat(1, batch_size, 1, 1, 1)

        outputs = snn_model(spike_data)
        probabilities = F.softmax(outputs.mean(dim=0), dim=-1)  # Moyenne et softmax pour obtenir des probabilités

        confidence, predicted = torch.max(probabilities, dim=-1)
        emotion = labels[predicted.item()]

        if confidence.item() > 0.4:
            results.append({
                'position': (x, y, w, h),
                'emotion': emotion,
                'confidence': confidence.item() * 1  # Multipliez par 100 pour des pourcentages
            })

    return results





