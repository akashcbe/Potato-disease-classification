import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = "potatoes.h5"
IMAGE_SIZE = 256
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Could not load model: {e}. Make sure potatoes.h5 is in the same folder.")
        st.stop()

def preprocess(image):
    img = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

st.set_page_config(page_title="Potato Disease Detector", page_icon="🥔", layout="centered")
st.title("🥔 Potato Disease Detector")
st.markdown("Upload a potato leaf image and the AI will diagnose it instantly.")
st.markdown("---")

model = load_model()

uploaded_file = st.file_uploader("Upload a potato leaf image (JPG or PNG)", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    with col2:
        with st.spinner("Analysing leaf..."):
            preds = model.predict(preprocess(image))[0]
            idx = int(np.argmax(preds))
            label = CLASS_NAMES[idx]
            confidence = float(preds[idx])
        icons = {"Early Blight":"🟠","Late Blight":"🔴","Healthy":"🟢"}
        st.markdown(f"### {icons[label]} {label}")
        st.markdown(f"**Confidence:** `{confidence*100:.1f}%`")
        st.progress(confidence)
    st.markdown("---")
    info = {"Early Blight": ("Caused by *Alternaria solani*. Dark brown spots with yellow rings.", "Apply fungicide early. Remove infected leaves."),
            "Late Blight":  ("Caused by *Phytophthora infestans*. Water-soaked lesions turning black.", "Act immediately. Use copper-based fungicide. Destroy infected plants."),
            "Healthy":      ("No disease detected. The leaf appears healthy.", "Maintain good watering and soil nutrition.")}
    st.info(info[label][0])
    st.success(f"**Recommended Action:** {info[label][1]}")
    with st.expander("📊 All prediction scores"):
        for cls, score in zip(CLASS_NAMES, preds):
            st.markdown(f"{icons[cls]} **{cls}**")
            st.progress(float(score), text=f"{score*100:.1f}%")
else:
    st.info("👆 Upload a leaf image above to get started.")
