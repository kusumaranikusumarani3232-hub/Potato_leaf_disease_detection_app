import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image

# -----------------------------
# LOAD MODEL
# -----------------------------
model = tf.keras.models.load_model(
    "potato_disease_model.h5"
)

# -----------------------------
# CLASS NAMES
# -----------------------------
class_names = [
    "Early Blight",
    "Late Blight",
    "Healthy"
]

# -----------------------------
# PAGE TITLE
# -----------------------------
st.title("🥔 Potato Leaf Disease Detection")

st.write(
    "Upload a potato leaf image to detect disease."
)

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "png", "jpeg"]
)

# -----------------------------
# PREDICTION
# -----------------------------
if uploaded_file is not None:

    # OPEN IMAGE
    image = Image.open(uploaded_file)

    # DISPLAY IMAGE
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # PREPROCESS IMAGE
    image = image.resize((128,128))

    image_array = np.array(image)

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # PREDICTION
    prediction = model.predict(image_array)

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    # OUTPUT
    st.success(
        f"Prediction: {class_names[predicted_class]}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )