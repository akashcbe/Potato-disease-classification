import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import struct

st.set_page_config(page_title="Potato Disease Classifier", page_icon="🥔", layout="wide")

IMAGE_SIZE = 256
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

DISEASE_INFO = {
    'Potato___Early_blight': {
        'name': 'Early Blight',
        'description': 'Early blight is a fungal disease caused by Alternaria solani. It typically affects older leaves first, causing dark, concentric rings that resemble a target.',
        'symptoms': ['Dark brown to black lesions on leaves', 'Concentric rings giving a "target" appearance', 'Yellowing of tissue around lesions'],
        'treatment': ['Apply fungicides containing chlorothalonil or mancozeb', 'Remove and destroy infected plant debris', 'Practice crop rotation (3-4 year cycles)'],
        'prevention': ['Use certified disease-free seed potatoes', 'Avoid overhead irrigation', 'Mulch around plants to reduce soil splash']
    },
    'Potato___Late_blight': {
        'name': 'Late Blight',
        'description': 'Late blight is a devastating disease caused by Phytophthora infestans. It was responsible for the Irish Potato Famine and can destroy entire crops rapidly.',
        'symptoms': ['Large, dark brown to black lesions on leaves', 'White, fuzzy growth on undersides of leaves', 'Rapid defoliation and plant collapse'],
        'treatment': ['Apply fungicides containing metalaxyl or cymoxanil', 'Remove and destroy infected plants immediately', 'Store tubers in cool, dry conditions'],
        'prevention': ['Use resistant potato varieties', 'Plant certified disease-free seed potatoes', 'Monitor weather conditions']
    },
    'Potato___healthy': {
        'name': 'Healthy',
        'description': 'Your potato plant appears healthy with no visible disease symptoms.',
        'symptoms': ['No visible lesions or spots on leaves', 'Green, vigorous foliage', 'Normal growth pattern'],
        'treatment': ['Continue regular monitoring for early disease signs', 'Maintain proper irrigation and fertilization'],
        'prevention': ['Continue current good agricultural practices', 'Regular scouting for early disease detection']
    }
}

def get_h5_path():
    candidates = [
        "potatoes.h5",
        os.path.join(os.path.dirname(__file__), "potatoes.h5"),
        "/mount/src/potato-disease-classification/potatoes.h5",
    ]
    if os.path.exists("/mount/src"):
        for folder in os.listdir("/mount/src"):
            candidates.append(f"/mount/src/{folder}/potatoes.h5")
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None

# ── Pure numpy CNN forward pass ───────────────────────────────────────────────

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def conv2d_valid(x, W, b):
    """x: (H,W,C), W: (kH,kW,C,F) -> (H-kH+1, W-kW+1, F)"""
    kH, kW, _, F = W.shape
    oH = x.shape[0] - kH + 1
    oW = x.shape[1] - kW + 1
    # Use stride tricks for speed
    out = np.zeros((oH, oW, F), dtype=np.float32)
    for f in range(F):
        Wf = W[:, :, :, f]  # (kH,kW,C)
        for i in range(oH):
            for j in range(oW):
                out[i, j, f] = np.sum(x[i:i+kH, j:j+kW, :] * Wf) + b[f]
    return out

def maxpool2d(x, size=2):
    H, W, C = x.shape
    oH, oW = H // size, W // size
    x_crop = x[:oH*size, :oW*size, :]
    x_r = x_crop.reshape(oH, size, oW, size, C)
    return x_r.max(axis=(1, 3))

def dense(x, W, b):
    return x @ W + b

@st.cache_resource
def load_model():
    import h5py
    path = get_h5_path()
    if path is None:
        st.error("potatoes.h5 not found!")
        return None

    weights = {}
    with h5py.File(path, "r") as f:
        def collect(name, obj):
            if isinstance(obj, h5py.Dataset):
                weights[name] = obj[()]
        f.visititems(collect)
    return weights

def find_weight(weights, layer_name, wtype):
    """Search for a weight by layer name and type (kernel/bias)."""
    for k, v in weights.items():
        parts = k.lower()
        if layer_name in parts and wtype in parts:
            return v
    return None

def run_inference(weights, img_arr):
    """Forward pass matching the CNN architecture in app.py original."""
    x = img_arr.copy()  # (256, 256, 3)

    # 6 Conv+Pool blocks
    conv_names = ['conv2d', 'conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4', 'conv2d_5']
    for name in conv_names:
        W = find_weight(weights, name, 'kernel')
        b = find_weight(weights, name, 'bias')
        if W is None or b is None:
            st.error(f"Missing weights for layer: {name}")
            return None
        x = relu(conv2d_valid(x, W, b))
        x = maxpool2d(x)

    x = x.flatten()

    # Dense layers
    W = find_weight(weights, 'dense', 'kernel')
    b = find_weight(weights, 'dense', 'bias')
    if W is None:
        st.error("Missing dense layer weights")
        return None
    x = relu(dense(x, W, b))

    W2 = find_weight(weights, 'dense_1', 'kernel')
    b2 = find_weight(weights, 'dense_1', 'bias')
    if W2 is None:
        st.error("Missing dense_1 layer weights")
        return None
    x = softmax(dense(x, W2, b2))
    return x

def preprocess_image(image):
    img = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    return np.array(img, dtype=np.float32) / 255.0

def predict(weights, image):
    img_arr = preprocess_image(image)
    probs = run_inference(weights, img_arr)
    if probs is None:
        return CLASS_NAMES[0], 0.0, np.array([1/3, 1/3, 1/3])
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs

def main():
    st.title("🥔 Potato Disease Detection System")

    with st.sidebar:
        st.header("📋 About")
        st.markdown("""
        Detects potato leaf diseases using Deep Learning:
        - **Early Blight** (Alternaria solani)
        - **Late Blight** (Phytophthora infestans)
        - **Healthy** leaves

        **How to use:**
        1. Upload a clear image of a potato leaf
        2. Click "Analyze Image"
        3. View the diagnosis
        """)
        st.divider()
        st.markdown("- **Model:** CNN (H5)\n- **Input:** 256×256 px\n- **Classes:** 3")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader("Choose a potato leaf image...", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Loading model and running inference..."):
                    weights = load_model()
                    if weights:
                        predicted_class, confidence, all_probs = predict(weights, image)
                        st.session_state.update({
                            'predicted_class': predicted_class,
                            'confidence': confidence,
                            'all_probs': all_probs,
                            'analyzed': True
                        })
                        st.rerun()

    with col2:
        st.subheader("📊 Results")
        if st.session_state.get('analyzed'):
            predicted_class = st.session_state['predicted_class']
            confidence = st.session_state['confidence']
            all_probs = st.session_state['all_probs']
            info = DISEASE_INFO[predicted_class]

            if predicted_class == 'Potato___healthy':
                st.success(f"### ✅ {info['name']}")
            elif 'Early_blight' in predicted_class:
                st.warning(f"### ⚠️ {info['name']}")
            else:
                st.error(f"### 🚨 {info['name']}")

            st.metric("Confidence Score", f"{confidence * 100:.2f}%")

            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ['#ff9800' if CLASS_NAMES[i] == predicted_class else '#90caf9' for i in range(3)]
            ax.bar(CLASS_NAMES, np.array(all_probs) * 100, color=colors)
            ax.set_ylabel('Probability (%)')
            ax.set_ylim(0, 100)
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
            plt.close()

            st.divider()
            st.subheader("📝 Diagnosis & Recommendations")
            st.markdown(f"**Description:** {info['description']}")
            with st.expander("🔍 Symptoms"):
                for s in info['symptoms']: st.markdown(f"- {s}")
            with st.expander("💊 Treatment"):
                for t in info['treatment']: st.markdown(f"- {t}")
            with st.expander("🛡️ Prevention"):
                for p in info['prevention']: st.markdown(f"- {p}")

            if confidence < 0.7:
                st.info("Low confidence — try a clearer, well-lit image of the leaf.")
        else:
            st.info("👈 Upload an image and click 'Analyze Image' to see results here")

    st.divider()
    st.markdown("<div style='text-align:center;color:#666;'>⚠️ For informational purposes only.</div>",
                unsafe_allow_html=True)

if __name__ == "__main__":
    main()
