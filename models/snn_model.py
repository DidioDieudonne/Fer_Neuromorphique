# models/snn_model.py

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from spikingjelly.activation_based import functional

# Fonction pour obtenir l'encodeur ResNet34 ajusté pour les poids en niveaux de gris (1 canal)
def get_encoder_snn(in_channels: int):
    resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    if in_channels == 1:
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    return resnet

class SNNModule(nn.Module):
    def __init__(self, in_channels, timesteps, n_classes):
        super(SNNModule, self).__init__()
        self.timesteps = timesteps
        self.encoder = get_encoder_snn(in_channels)
        self.fc = nn.Linear(1000, n_classes, bias=False)

    def forward(self, x):
        if x.dim() == 5:
            outputs = [self.encoder(x[:, t]) for t in range(x.shape[1])]
            x = torch.stack(outputs).mean(0)
        else:
            raise ValueError(f"Unexpected data shape: {x.shape}")

        functional.reset_net(self.encoder)
        x = torch.flatten(x, 1)
        return self.fc(x)

# Fonction pour charger le modèle SNN avec les poids sauvegardés
def load_snn_model(model_path="models/snn_model_p.pth"):
    in_channels = 1  # Canal d'entrée, ici en niveaux de gris
    timesteps = 5  # Nombre de pas de temps
    n_classes = 7  # Nombre de classes de sortie pour la classification des émotions
    snn_model = SNNModule(in_channels, timesteps, n_classes)
    
    # Charger les poids avec `map_location` pour les CPU et `strict=False` pour ignorer les erreurs mineures
    snn_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
    snn_model.eval()
    print("Modèle SNN chargé avec succès.")
    return snn_model





