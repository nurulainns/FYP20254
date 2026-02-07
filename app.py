import streamlit as st
import numpy as np
import pickle
import cv2
from skimage.feature import hog

# --- Configuration ---
# Ensure these filenames match exactly what you upload to Hugging Face
IMAGE_SIZE = 256
CELLS_PER_BLOCK = (2, 2)
PIXELS_PER_CELL = (16, 16)
ORIENTATIONS = 9
NORMALISE_HOG = 'L2-Hys'
CLASS_LABELS = {0: "Healthy", 1: "Parkinson's Disease"}

MODEL_PATH = "svm_hog_model_8020.pkl"
SCALER_PATH = "hog_features_split_scaled_8020.npz"

# --- Page Config ---
st.set_page_config(
    page_title="Parkinson's Detection AI",
    page_icon="🧠",
    layout="centered"
)

# --- Title ---
st.title("🧠 Parkinson’s Disease Detection Based on Spiral Handwriting 🌀")
st.caption("Upload a spiral drawing to analyze handwriting patterns for detecting Parkinson's Disease using SVM + HOG.")

# --- Load Model & Scaler ---
@st.cache_resource
def load_assets():
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        data = np.load(SCALER_PATH, allow_pickle=True)
        scaler = data["scaler"][()]
        return model, scaler
    except FileNotFoundError:
        st.error(f"Error: Model or Scaler file not found. Please ensure {MODEL_PATH} and {SCALER_PATH} are uploaded.")
        return None, None

model_svm, scaler = load_assets()

# --- Helper Function ---
def preprocess_and_extract_hog(image_file):
    # Read image from Streamlit upload
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Invalid image file format.")

    # Preprocessing
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    # Feature Extraction
    features = hog(
        image,
        orientations=ORIENTATIONS,
        pixels_per_cell=PIXELS_PER_CELL,
        cells_per_block=CELLS_PER_BLOCK,
        block_norm=NORMALISE_HOG,
        feature_vector=True
    )

    return features.reshape(1, -1)

# --- Image Upload ---
# --- Image Upload ---
uploaded_file = st.file_uploader(
    "  ", # This is your label
    type=["png", "jpg", "jpeg"]
)

# --- Logic on Upload ---
if uploaded_file and model_svm is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing handwriting pattern..."):
        try:
            # 1. Extract Features
            features = preprocess_and_extract_hog(uploaded_file)
            
            # 2. Scale Features
            scaled_features = scaler.transform(features)

            # 3. Predict
            prediction = model_svm.predict(scaled_features)[0]
            confidence = model_svm.decision_function(scaled_features)[0]

            # 4. Display Result
            st.divider()
            st.subheader("Analysis Result")

            if prediction == 1:
                st.error(f"**Result:** {CLASS_LABELS[prediction]}")
            else:
                st.success(f"**Result:** {CLASS_LABELS[prediction]}")

            # Metric for distance from decision boundary
            # st.metric("Confidence Score", f"{confidence:.4f}")
            # st.info("The confidence score reflects the distance from the SVM decision boundary.")

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")