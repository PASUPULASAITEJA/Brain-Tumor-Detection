import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI scan to check whether a tumor is present.")

# Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/brain_tumor_model.h5")

# Image preprocessing
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((150,150))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file)
    except:
        st.error("❌ Invalid image file.")
        st.stop()

    # Show uploaded image
    st.image(image, caption="Uploaded MRI Image", width=300)

    with st.spinner("Analyzing MRI..."):

        try:
            model = load_model()
        except:
            st.error("❌ Model file not found. Please run train_model.py first.")
            st.stop()

        processed_image = preprocess_image(image)

        prediction = model.predict(processed_image)[0][0]

        if prediction >= 0.5:
            st.error("🔴 Tumor Detected")
            confidence = prediction
        else:
            st.success("🟢 No Tumor Detected")
            confidence = 1 - prediction

        st.write(f"Confidence: {confidence*100:.2f}%")

else:
    st.info("👆 Please upload an MRI image to start detection.")

st.caption("⚠️ This system is for educational purposes only and not a medical diagnosis.")