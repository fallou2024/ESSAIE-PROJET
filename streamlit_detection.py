import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing import image
from PIL import Image

# Charger le modèle entraîné
MODEL_PATH = "modele_panneaux.h5"
st.title("Détection des Panneaux de Signalisation")
st.write("Cette application permet de reconnaître les panneaux de signalisation à partir d'images ou d'une webcam.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Charger les classes
DATA_DIR = "c:/Users/toshiba/OneDrive/Bureau/NAAL_GOMYCODE/IMAGE_PANNEAU"
class_labels = sorted(os.listdir(DATA_DIR))

def predict_image(img):
    img = img.resize((64, 64))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_class, confidence

# Upload d'image
uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Image chargée", use_column_width=True)
    predicted_class, confidence = predict_image(img)
    st.write(f"**Panneau détecté : {predicted_class} ({confidence:.2f})**")

# Détection via webcam
if st.button("Activer la webcam"):
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Erreur: Impossible d'accéder à la webcam")
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        predicted_class, confidence = predict_image(img_pil)
        cv2.putText(frame, f"{predicted_class} ({confidence:.2f})", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        stframe.image(frame, channels="BGR")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
