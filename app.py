import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="Marine Plastic Detection",
    page_icon="🌊",
    layout="centered"
)


# -----------------------------
# Title
# -----------------------------

st.title("🌊 Marine Plastic Detection")

st.write(
    "Upload an image to detect whether it contains "
    "marine animals or plastic waste."
)


# -----------------------------
# Load model
# -----------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "marine_plastic_model.keras"
    )


model = load_model()


# -----------------------------
# Upload image
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


# -----------------------------
# Prediction
# -----------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🔍 Detect"):

        # Resize exactly like training
        img = image.resize((64, 64))

        # Convert to numpy
        img_array = np.array(
            img,
            dtype=np.float32
        )

        # Same scaling used during training
        img_array = img_array / 255.0

        # Add batch dimension
        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        # Prediction
        prediction = model.predict(
            img_array,
            verbose=0
        )[0][0]


        # --------------------------------
        # CHANGE THIS BASED ON class_indices
        # --------------------------------

        if prediction >= 0.5:

            result = "Plastic Waste"
            confidence = prediction

        else:

            result = "Marine Animals"
            confidence = 1 - prediction


        # Display result
        st.success(
            f"Prediction: {result}"
        )

        st.write(
            f"Confidence: {confidence * 100:.2f}%"
        )

        st.progress(
            float(confidence)
        )