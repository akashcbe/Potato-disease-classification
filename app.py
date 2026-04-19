# app.py
import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import h5py
import json

# Page configuration
st.set_page_config(
    page_title="Potato Disease Classifier",
    page_icon="🥔",
    layout="wide"
)

# Constants
IMAGE_SIZE = 256
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Disease information dictionary
DISEASE_INFO = {
    'Potato___Early_blight': {
        'name': 'Early Blight',
        'description': 'Early blight is a fungal disease caused by Alternaria solani. It typically affects older leaves first, causing dark, concentric rings that resemble a target.',
        'symptoms': [
            'Dark brown to black lesions on leaves',
            'Concentric rings giving a "target" appearance',
            'Yellowing of tissue around lesions',
            'Lesions may also appear on stems and tubers'
        ],
        'treatment': [
            'Apply fungicides containing chlorothalonil or mancozeb',
            'Remove and destroy infected plant debris',
            'Practice crop rotation (3-4 year cycles)',
            'Ensure proper plant spacing for air circulation',
            'Water at base of plants to keep foliage dry'
        ],
        'prevention': [
            'Use certified disease-free seed potatoes',
            'Avoid overhead irrigation',
            'Mulch around plants to reduce soil splash',
            'Apply preventive fungicides in high-risk conditions'
        ]
    },
    'Potato___Late_blight': {
        'name': 'Late Blight',
        'description': 'Late blight is a devastating disease caused by Phytophthora infestans. It was responsible for the Irish Potato Famine and can destroy entire crops rapidly.',
        'symptoms': [
            'Large, dark brown to black lesions on leaves',
            'White, fuzzy growth on undersides of leaves in humid conditions',
            'Stems show dark brown to black lesions',
            'Tubers develop dark, shrunken areas on the skin',
            'Rapid defoliation and plant collapse'
        ],
        'treatment': [
            'Apply fungicides containing metalaxyl or cymoxanil',
            'Remove and destroy infected plants immediately',
            'Harvest healthy tubers promptly',
            'Store tubers in cool, dry conditions'
        ],
        'prevention': [
            'Use resistant potato varieties',
            'Plant certified disease-free seed potatoes',
            'Destroy volunteer potatoes and cull piles',
            'Apply preventive fungicides before disease appears',
            'Monitor weather conditions (cool, wet weather favors disease)'
        ]
    },
    'Potato___healthy': {
        'name': 'Healthy',
        'description': 'Your potato plant appears healthy with no visible disease symptoms.',
        'symptoms': [
            'No visible lesions or spots on leaves',
            'Green, vigorous foliage',
            'Normal growth pattern'
        ],
        'treatment': [
            'Continue regular monitoring for early disease signs',
            'Maintain proper irrigation and fertilization',
            'Practice good field sanitation',
            'Rotate crops regularly'
        ],
        'prevention': [
            'Continue current good agricultural practices',
            'Regular scouting for early disease detection',
            'Maintain proper plant spacing'
        ]
    }
}

# ── Minimal numpy-only CNN inference ──────────────────────────────────────────

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def conv2d(x, W, b):
    """x: HWC, W: h,w,in,out"""
    h, w, _ = x.shape
    kh, kw, _, out_c = W.shape
    oh, ow = h - kh + 1, w - kw + 1
    out = np.zeros((oh, ow, out_c), dtype=np.float32)
    for f in range(out_c):
        for i in range(oh):
            for j in range(ow):
                out[i, j, f] = np.sum(x[i:i+kh, j:j+kw, :] * W[:,:,:,f]) + b[f]
    return out

def maxpool2d(x, size=2):
    h, w, c = x.shape
    oh, ow = h // size, w // size
    out = np.zeros((oh, ow, c), dtype=np.float32)
    for i in range(oh):
        for j in range(ow):
            out[i, j, :] = x[i*size:(i+1)*size, j*size:(j+1)*size, :].max(axis=(0,1))
    return out

def dense(x, W, b):
    return x @ W + b

@st.cache_resource
def load_weights():
    """Load weights from the .h5 file."""
    try:
        weights = {}
        with h5py.File("potatoes.h5", "r") as f:
            def collect(name, obj):
                if isinstance(obj, h5py.Dataset):
                    weights[name] = obj[()]
            f.visititems(collect)
        return weights
    except Exception as e:
        return None

def extract_layer_weights(weights, layer_name):
    """Find kernel and bias for a layer by searching keys."""
    kernel, bias = None, None
    for k, v in weights.items():
        if layer_name in k and 'kernel' in k:
            kernel = v
        if layer_name in k and 'bias' in k:
            bias = v
    return kernel, bias

@st.cache_resource
def load_model():
    """Load the ONNX model if available, else fall back to h5 weights."""
    # Try ONNX first
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession("potatoes.onnx")
        return ('onnx', sess)
    except Exception:
        pass

    # Try loading h5 weights directly
    weights = load_weights()
    if weights is not None:
        return ('h5', weights)

    return None

def preprocess_image(image):
    img = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr

def predict_onnx(sess, img_arr):
    inp_name = sess.get_inputs()[0].name
    batch = np.expand_dims(img_arr, 0)
    preds = sess.run(None, {inp_name: batch})[0][0]
    return preds

def predict_h5(weights, img_arr):
    """Run forward pass using extracted h5 weights. 
    Works if the model architecture matches the standard CNN."""
    # Build layer name list matching Keras sequential layer naming
    conv_layers = [f'conv2d' if i == 0 else f'conv2d_{i}' for i in range(6)]
    
    x = img_arr.copy()
    for i, lname in enumerate(conv_layers):
        W, b = extract_layer_weights(weights, lname)
        if W is None:
            st.error(f"Could not find weights for layer: {lname}")
            return np.array([1/3, 1/3, 1/3])
        x = relu(conv2d(x, W, b))
        x = maxpool2d(x)

    x = x.flatten()

    W, b = extract_layer_weights(weights, 'dense')
    x = relu(dense(x, W, b))

    W, b = extract_layer_weights(weights, 'dense_1')
    x = softmax(dense(x, W, b))
    return x

def predict(model, image):
    img_arr = preprocess_image(image)
    mode, obj = model

    if mode == 'onnx':
        probs = predict_onnx(obj, img_arr)
    else:
        with st.spinner("Running inference (first run may be slow)..."):
            probs = predict_h5(obj, img_arr)

    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs

# ── UI ─────────────────────────────────────────────────────────────────────────

def main():
    st.title("🥔 Potato Disease Detection System")

    with st.sidebar:
        st.header("📋 About")
        st.markdown("""
        This system uses Deep Learning to detect potato leaf diseases:
        - **Early Blight** (Alternaria solani)
        - **Late Blight** (Phytophthora infestans)
        - **Healthy** leaves

        **How to use:**
        1. Upload a clear image of a potato leaf
        2. Click "Analyze Image"
        3. View the diagnosis and recommendations
        """)
        st.divider()
        st.header("📊 Model Information")
        st.markdown("""
        - **Model Type:** CNN
        - **Image Size:** 256×256 px
        - **Classes:** 3
        """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a potato leaf image...",
            type=['jpg', 'jpeg', 'png']
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                model = load_model()
                if model is None:
                    st.error("Could not load model. Make sure potatoes.h5 or potatoes.onnx is in the repo root.")
                else:
                    predicted_class, confidence, all_probs = predict(model, image)
                    st.session_state.update({
                        'predicted_class': predicted_class,
                        'confidence': confidence,
                        'all_probs': all_probs,
                        'analyzed': True
                    })
                    st.rerun()

    with col2:
        st.subheader("📊 Results")
        if st.session_state.get('analyzed', False):
            predicted_class = st.session_state['predicted_class']
            confidence = st.session_state['confidence']
            all_probs = st.session_state['all_probs']

            if predicted_class == 'Potato___healthy':
                st.success(f"### ✅ {DISEASE_INFO[predicted_class]['name']}")
            elif 'Early_blight' in predicted_class:
                st.warning(f"### ⚠️ {DISEASE_INFO[predicted_class]['name']}")
            else:
                st.error(f"### 🚨 {DISEASE_INFO[predicted_class]['name']}")

            st.metric("Confidence Score", f"{confidence * 100:.2f}%")

            st.subheader("📈 Prediction Probabilities")
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ['#4caf50' if CLASS_NAMES[i] == predicted_class else '#ff6b6b'
                      for i in range(len(CLASS_NAMES))]
            ax.bar(CLASS_NAMES, np.array(all_probs) * 100, color=colors)
            ax.set_ylabel('Probability (%)')
            ax.set_title('Disease Classification Probabilities')
            ax.set_ylim(0, 100)
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
            plt.close()

            st.divider()
            st.subheader("📝 Diagnosis & Recommendations")
            info = DISEASE_INFO[predicted_class]
            st.markdown(f"**Description:** {info['description']}")

            with st.expander("🔍 Common Symptoms"):
                for s in info['symptoms']:
                    st.markdown(f"- {s}")
            with st.expander("💊 Recommended Treatment"):
                for t in info['treatment']:
                    st.markdown(f"- {t}")
            with st.expander("🛡️ Prevention Tips"):
                for p in info['prevention']:
                    st.markdown(f"- {p}")

            if confidence < 0.7:
                st.info("Confidence is moderate. Try a clearer, well-lit photo of the leaf.")
        else:
            st.info("👈 Upload an image and click 'Analyze Image' to see results here")

    st.divider()
    st.markdown("""
    <div style='text-align:center;color:#666;padding:1rem;'>
        <p>⚠️ For informational purposes only. Consult agricultural experts for definitive diagnosis.</p>
        <p>Made with ❤️ using Python & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
