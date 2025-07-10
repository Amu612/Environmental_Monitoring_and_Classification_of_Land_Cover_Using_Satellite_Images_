import streamlit as st
import numpy as np
from PIL import Image
import requests
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Download model from Google Drive only once
model_url = "https://drive.google.com/uc?export=download&id=1QCoEhMnBGkA14vepBhM0gpy3PeSsjAdb"
model_path = "Modelenv.v1.h5"

if not os.path.exists(model_path):
    with st.spinner("Downloading model..."):
        r = requests.get(model_url)
        with open(model_path, "wb") as f:
            f.write(r.content)

# Load the model
model = load_model(model_path)

# Define class labels
labels = {
    0: "Cloudy",
    1: "Desert",
    2: "Green_Area",
    3: "Water"
}

# Streamlit App UI
st.set_page_config(page_title="Satellite Image Classifier", layout="centered")
st.title("🛰️ Land Cover Classification from Satellite Images")
st.markdown("Upload a satellite image and classify it as one of the following:")
st.markdown("- 🌥️ Cloudy\n- 🏜️ Desert\n- 🌳 Green Area\n- 💧 Water")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_data = Image.open(uploaded_file).convert("RGB")
    st.image(image_data, caption="Uploaded Image", use_column_width=True)

    # Preprocessing
    img = image_data.resize((256, 256))  # Match model input
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])
    class_label = labels[predicted_class]

    st.success(f"🔍 **Prediction:** {class_label}")
