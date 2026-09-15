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
# Background Image Setup
# -----------------------------

def set_background(url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{url}");
            background-attachment: fixed;
            background-size: cover;
            background-position: center;
        }}
        /* Overlay card styling for readability */
        .block-container {{
            background-color: rgba(255, 255, 255, 0.90);
            padding: 2rem;
            border-radius: 12px;
            margin-top: 2rem;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Ocean background URL (Replace with any public image URL or local base64 string)
background_image_url = "https://images.unsplash.com/photo-1507525428034-b723cf961d3e"
set_background(background_image_url)


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
        if result == "Plastic Waste":
            st.error(f"Prediction: {result}")
        else:
            st.success(f"Prediction: {result}")

        st.write(
            f"Confidence: {confidence * 100:.2f}%"
        )

        st.progress(
            float(confidence)
        )

        # ---------------------------------------------
        # Actionable Solutions (Only for Plastic Waste)
        # ---------------------------------------------
        if result == "Plastic Waste":
            st.markdown("---")
            st.subheader("🚨 Action Required: Plastic Waste Mitigation Solutions")
            
            st.markdown("""
            **1. Immediate Clean-up & Reporting**
            * **Report Location:** Geo-tag and report the location to local marine conservation authorities or coastal cleanup NGOs.
            * **Safe Manual Removal:** If accessible and safe, collect macro-plastics using debris nets to prevent ingestion by marine life.

            **2. Disposal & Material Recovery**
            * **Segregation:** Separate recyclable polymers (PET/HDPE) from bio-contaminated or degraded debris.
            * **Mechanical Recycling:** Route rigid plastics to specialized recycling facilities for downcycling or pelletization.
            * **Energy Recovery:** Send non-recyclable ocean-bound plastic to specialized waste-to-energy facilities equipped with scrubbers.

            **3. Preventive & Long-term Measures**
            * **Upstream Interception:** Install river barriers, trash wheels, or interceptor boom systems at nearby river mouths.
            * **Community Education:** Support single-use plastic bans in coastal zones and promote circular packaging alternatives.
            """)