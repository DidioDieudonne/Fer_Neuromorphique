
import numpy as np

# Dictionnaire des étiquettes d'émotions
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Fonction pour extraire les caractéristiques de l'image
def extract_features(image):
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    if len(image.shape) == 2:
        feature = image.reshape(1, 48, 48)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        feature = image.reshape(1, 48, 48)
    else:
        raise ValueError("L'image doit être en niveaux de gris avec une forme (48, 48) ou (48, 48, 1)")

    feature = feature / 255.0
    return feature

