#  Potato Disease Classification System

A deep learning project that detects diseases in potato leaves using a Convolutional Neural Network (CNN). Just upload a leaf image and the model will tell you if the plant is **Healthy**, has **Early Blight**, or **Late Blight**.

---

##  Overview

Potato diseases can destroy entire crops if not detected early. This project uses a trained CNN model to automatically classify potato leaf images into one of three categories. It also provides a REST API (built with FastAPI) so the model can be used in other apps or websites.

This was built as a student learning project to explore deep learning and API development.

---

##  Features

- Classifies potato leaf images into 3 categories:
  - ✅ Healthy
  - 🟠 Early Blight *(caused by Alternaria solani)*
  - 🔴 Late Blight *(caused by Phytophthora infestans)*
- REST API for sending images and getting predictions
- Returns predicted class and confidence score
- Interactive web UI built with Streamlit
- Trained on the PlantVillage dataset

---

##  Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10 | Programming language |
| TensorFlow / Keras | Building and training the CNN model |
| FastAPI | REST API for predictions |
| Uvicorn | ASGI server to run FastAPI |
| NumPy | Image array processing |
| Pillow (PIL) | Reading and handling images |
| Streamlit | Interactive web UI |

---

##  Folder Structure

```
potato-disease-classification/
│
├── main.py                 # FastAPI backend — prediction API
├── app.py                  # Streamlit frontend — web UI
├── requirements.txt        # All Python dependencies
│
├── saved_models/
│   └── 1/                  # Saved trained model (TensorFlow SavedModel format)
│
└── dataset/                # Training images (PlantVillage dataset)
    ├── Potato___Early_blight/
    ├── Potato___Late_blight/
    └── Potato___healthy/
```

---

##  Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/potato-disease-classification.git
cd potato-disease-classification
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# or
source venv/bin/activate     # Mac / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

##  How to Run

### Option A — Streamlit Web App

```bash
streamlit run app.py
```

Open your browser at: `http://localhost:8501`

Upload a potato leaf image and click **Analyze** to get a prediction.

---

### Option B — FastAPI Backend

```bash
python main.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --reload
```

API will be live at: `http://localhost:8000`

Explore the auto-generated docs at: `http://localhost:8000/docs`

---

## 🔌 API Usage

### Health Check

```
GET /ping
```

**Response:**
```
"Hello, I am alive"
```

---

### Predict Disease

```
POST /predict
```

Send a leaf image as `multipart/form-data`:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@potato_leaf.jpg"
```

**Response:**
```json
{
  "class": "Early Blight",
  "confidence": 0.9732
}
```

---

##  Future Improvements

- [ ] Add support for more crops (tomato, corn, etc.)
- [ ] Deploy the API to the cloud (AWS / GCP / Render)
- [ ] Build a mobile-friendly frontend
- [ ] Improve model accuracy with data augmentation
- [ ] Add a map to track disease spread by region

---

##  Author

**Akash**
- 🎓 Student Project
- GitHub: [akashcbe](https://github.com/akashcbe/Potato-disease-classification)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

Screenshot
![image alt](https://github.com/akashcbe/Potato-disease-classification/blob/f38ec267c73629daa26e308564ae340e60ae8407/Screenshot%202026-04-18%20141716.png)

![image alt](https://github.com/akashcbe/Potato-disease-classification/blob/e938c97dc829b0e871bbf5e7dd22d86b9c282f35/Screenshot%202026-04-18%20141736.png)

![image alt](https://github.com/akashcbe/Potato-disease-classification/blob/3f0c97b765dd1219b75a6905f99784e4ed099ce2/Screenshot%202026-04-18%20141756.png)
