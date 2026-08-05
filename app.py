#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image




st.set_page_config(
    page_title="Fruit Quality Detection",
    page_icon="🍎",
    layout="centered"
)

# ======================================
# Configuration
# ======================================

MODEL_PATH = "models/fruit_quality_model.h5"
TRAIN_DIR = "dataset/train"

IMG_SIZE = (224, 224)

# ======================================
# Load Model
# ======================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ======================================
# Load Class Names
# ======================================

class_names = sorted([
    folder
    for folder in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, folder))
])

# ======================================
# Prediction Function
# ======================================

def predict(img):

    img = img.resize(IMG_SIZE)

    img_array = np.array(img)

    # Convert grayscale to RGB if needed
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)

    # Remove alpha channel if present
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = np.expand_dims(img_array, axis=0)

    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    prediction = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(prediction)

    confidence = prediction[0][predicted_index]

    return prediction[0], predicted_index, confidence

# ======================================
# Streamlit UI
# ======================================


st.title("🍎 Fruit Quality Detection")

st.markdown("""
Upload a fruit image to classify its quality using a MobileNetV2 deep learning model.
""")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image")

    with st.spinner("Analyzing image..."):

        predictions, idx, confidence = predict(image)

       
    st.success("Prediction Complete!")

    st.subheader("Prediction")

    st.write(f"**Class:** {class_names[idx]}")

    st.write(f"**Confidence:** {confidence*100:.2f}%")

    st.subheader("Top Predictions")

    top3 = np.argsort(predictions)[::-1][:3]

    for i in top3:

        st.progress(float(predictions[i]))

        st.write(
            f"{class_names[i]} : {predictions[i]*100:.2f}%"
        )

