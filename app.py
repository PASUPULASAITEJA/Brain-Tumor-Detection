import time
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = 224

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI scan to check whether a tumor is present.")

@st.cache_resource
def load_model() -> tf.keras.Model:
    return tf.keras.models.load_model("model/brain_tumor_model.h5")

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
    except Exception as e:
        st.error(f"❌ Invalid image file: {e}")
        st.stop()

    st.image(image, caption="Uploaded MRI Image", width=350)
    
    analyze_button = st.button("Analyze Image", use_container_width=True, type="primary")

    if analyze_button:
        with st.spinner("Analyzing MRI..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
                
            try:
                model = load_model()
            except Exception as e:
                st.error("❌ Model file not found. Please run `python train_model.py` first.")
                st.stop()

            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image, verbose=0)[0][0]

            st.divider()

            if prediction >= 0.5:
                st.error("### 🔴 Tumor Detected")
                confidence = prediction
            else:
                st.success("### 🟢 No Tumor Detected")
                confidence = 1 - prediction

            st.write(f"**Confidence:** {confidence*100:.2f}%")
            st.progress(float(confidence))

else:
    st.info("👆 Please upload an MRI image and click 'Analyze Image' to start detection.")

st.caption("⚠️ **Disclaimer**: This system is for educational purposes only and should not be used for medical diagnosis.")