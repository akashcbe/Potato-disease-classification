import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Potato Disease Classifier", page_icon="🥔", layout="wide")

IMAGE_SIZE = 256
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

DISEASE_INFO = {
    'Potato___Early_blight': {
        'name': 'Early Blight',
        'description': 'Early blight is a fungal disease caused by Alternaria solani. It typically affects older leaves first, causing dark, concentric rings that resemble a target.',
        'symptoms': ['Dark brown to black lesions on leaves', 'Concentric rings giving a "target" appearance', 'Yellowing of tissue around lesions', 'Lesions may also appear on stems and tubers'],
        'treatment': ['Apply fungicides containing chlorothalonil or mancozeb', 'Remove and destroy infected plant debris', 'Practice crop rotation (3-4 year cycles)', 'Ensure proper plant spacing for air circulation'],
        'prevention': ['Use certified disease-free seed potatoes', 'Avoid overhead irrigation', 'Mulch around plants to reduce soil splash']
    },
    'Potato___Late_blight': {
        'name': 'Late Blight',
        'description': 'Late blight is a devastating disease caused by Phytophthora infestans. It was responsible for the Irish Potato Famine and can destroy entire crops rapidly.',
        'symptoms': ['Large, dark brown to black lesions on leaves', 'White, fuzzy growth on undersides of leaves in humid conditions', 'Rapid defoliation and plant collapse'],
        'treatment': ['Apply fungicides containing metalaxyl or cymoxanil', 'Remove and destroy infected plants immediately', 'Store tubers in cool, dry conditions'],
        'prevention': ['Use resistant potato varieties', 'Plant certified disease-free seed potatoes', 'Monitor weather conditions (cool, wet weather favors disease)']
    },
    'Potato___healthy': {
        'name': 'Healthy',
        'description': 'Your potato plant appears healthy with no visible disease symptoms.',
        'symptoms': ['No visible lesions or spots on leaves', 'Green, vigorous foliage', 'Normal growth pattern'],
        'treatment': ['Continue regular monitoring for early disease signs', 'Maintain proper irrigation and fertilization'],
        'prevention': ['Continue current good agricultural practices', 'Regular scouting for early disease detection']
    }
}

@st.cache_resource
def load_model():
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession("potatoes.onnx")
        return sess
    except Exception as e:
        st.error(f"Could not load model: {e}\n\nMake sure potatoes.onnx is in the repo root.")
        return None

def preprocess_image(image):
    img = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)

def predict(sess, image):
    batch = preprocess_image(image)
    inp_name = sess.get_inputs()[0].name
    probs = sess.run(None, {inp_name: batch})[0][0]
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
        3. View the diagnosis and recommendations
        """)
        st.divider()
        st.header("📊 Model Info")
        st.markdown("- **Type:** CNN (ONNX)\n- **Input:** 256×256 px\n- **Classes:** 3")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader("Choose a potato leaf image...", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                sess = load_model()
                if sess:
                    predicted_class, confidence, all_probs = predict(sess, image)
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

            st.subheader("📈 Prediction Probabilities")
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
    st.markdown("<div style='text-align:center;color:#666;'>⚠️ For informational purposes only. Consult agricultural experts for definitive diagnosis.</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
