import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Set page config
st.set_page_config(page_title="Satellite Image Classifier", layout="centered")

# Title
st.title("🌍 Satellite Image Classification")
st.write("Upload an image and let the model classify it as Cloudy, Desert, Green Area, or Water.")

# Function to load/download model
@st.cache_resource
def load_model():
    model_path = "Modelenv.v1.h5"
    if not os.path.exists(model_path):
        # Download from Google Drive
        url = "https://drive.google.com/uc?export=download&id=1YCmlwGcV53n-C9sij4JlLoJEfavXwo6b"
        gdown.download(url, model_path, quiet=False)
    model = tf.keras.models.load_model(model_path)
    return model

# Load the model
model = load_model()

# Class labels
class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']

# Upload image
uploaded_file = st.file_uploader("Upload Satellite Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)


    # Preprocess image
    img = image.resize((256, 256))
    img_array = np.array(img)

    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]  # remove alpha channel

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_label = class_names[predicted_index]

    st.success(f"✅ Prediction: **{predicted_label}**")
    st.write("📊 Confidence Scores:")
    for name, score in zip(class_names, predictions[0]):
        st.write(f"- {name}: {score:.2%}")
