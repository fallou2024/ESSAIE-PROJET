import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import tempfile

# Chemin vers le modèle et dataset
MODEL_PATH = "c:/Users/toshiba/OneDrive/Bureau/NAAL_GOMYCODE/IMAGE_PANNEAU/modele_panneaux.h5"
DATASET_PATH = "c:/Users/toshiba/OneDrive/Bureau/NAAL_GOMYCODE/IMAGE_PANNEAU/traffic_Data"

# Charger le modèle
@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_trained_model()

# Récupérer les noms des classes
class_names = sorted(os.listdir(DATASET_PATH))

# Fonction de prédiction
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    
    return class_names[predicted_class], confidence

# Interface Streamlit
st.title("Détection des Panneaux de Signalisation 🚦")

# Option d'upload d'image
uploaded_file = st.file_uploader("Télécharge une image de panneau", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_file.write(uploaded_file.read())
    temp_file_path = temp_file.name

    # Affichage de l'image uploadée
    st.image(temp_file_path, caption="Image chargée", use_column_width=True)

    # Prédiction
    predicted_label, confidence = predict_image(temp_file_path)

    # Résultat
    st.subheader(f"🔍 Panneau détecté : {predicted_label} ({confidence * 100:.2f}%)")
