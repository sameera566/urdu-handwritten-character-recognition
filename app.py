
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests
import json
from io import BytesIO

st.set_page_config(
    page_title="Urdu AI Studio",
    page_icon="🧠",
    layout="wide"
)

MODEL_URL = "https://huggingface.co/sameera123-2026/urdu-handwritten-character-recognition/resolve/main/model.pth"
LABELS_URL = "https://huggingface.co/sameera123-2026/urdu-handwritten-character-recognition/resolve/main/labels.json"

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #050719, #10082b, #080d25);
    color: white;
}
header {visibility: hidden;}
.block-container {
    padding-top: 1.5rem;
    max-width: 1250px;
}
h1, h2, h3, p, label {
    color: white !important;
}
.hero {
    padding: 30px 10px;
    margin-bottom: 20px;
}
.hero h1 {
    font-size: 42px;
    font-weight: 800;
}
.hero span {
    color: #a855f7;
}
.subtitle {
    color: #c4b5fd !important;
    font-size: 18px;
}
.panel {
    background: linear-gradient(145deg, #19143b, #0d1230);
    border: 1px solid #6535b8;
    border-radius: 22px;
    padding: 25px;
    min-height: 330px;
}
.result {
    background: linear-gradient(145deg, #211052, #100b2b);
    border: 1px solid #9333ea;
    border-radius: 18px;
    padding: 25px;
    text-align: center;
}
.character {
    font-size: 100px;
    color: white;
    font-family: serif;
}
.metric {
    background: #171438;
    border: 1px solid #493078;
    border-radius: 15px;
    padding: 18px;
    text-align: center;
}
.metric h2 {
    color: #c084fc !important;
    margin-bottom: 3px;
}
.stButton button {
    background: linear-gradient(90deg, #9333ea, #4f46e5);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: bold;
    height: 48px;
    width: 100%;
}
.stButton button:hover {
    border: 1px solid #c084fc;
    color: white;
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    response = requests.get(MODEL_URL, timeout=120)
    response.raise_for_status()

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 42)
    model.load_state_dict(
        torch.load(BytesIO(response.content), map_location="cpu")
    )
    model.eval()
    return model

@st.cache_data
def load_labels():
    response = requests.get(LABELS_URL, timeout=60)
    response.raise_for_status()
    return json.loads(response.text)

def predict(image, model, labels):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = transform(image.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        score, index = torch.max(probabilities, 1)

    return labels[index.item()], score.item() * 100

# Header
st.markdown("""
<div class="hero">
    <h1>🧠 Urdu AI <span>Studio</span></h1>
    <p class="subtitle">
        Smart Urdu Handwritten Character Recognition
    </p>
</div>
""", unsafe_allow_html=True)

# Navigation
nav1, nav2, nav3 = st.columns(3)
with nav1:
    st.markdown("**🏠 Home**")
with nav2:
    st.markdown("**📊 Model Information**")
with nav3:
    st.markdown("**⚡ Powered by ResNet18**")

st.divider()

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("""
    <div class="panel">
        <h2>📤 Upload Your Character</h2>
        <p>Choose an image containing one Urdu character.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload character image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Character", width=180)

        recognize = st.button(
            "✨ Recognize Character",
            use_container_width=True
        )
    else:
        recognize = False
        st.info("Please upload a single Urdu character image.")

with right:
    st.markdown("""
    <div class="panel">
        <h2>🤖 Prediction Result</h2>
        <p>Your AI recognition result will appear here.</p>
    </div>
    """, unsafe_allow_html=True)

    if uploaded_file and recognize:
        try:
            with st.spinner("AI is recognizing your character..."):
                model = load_model()
                labels = load_labels()
                character, confidence = predict(image, model, labels)

            st.markdown(f"""
            <div class="result">
                <p>Predicted Urdu Character</p>
                <div class="character">{character}</div>
                <h3>Recognition Complete ✓</h3>
                <h2>{confidence:.2f}%</h2>
                <p>Model confidence score</p>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")

    else:
        st.markdown("""
        <div class="result">
            <div class="character">؟</div>
            <h3>Waiting for Image</h3>
            <p>Upload an image and click Recognize Character.</p>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# Project statistics
st.markdown("### 📊 Project Overview")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class="metric">
        <h2>19,901</h2>
        <p>Dataset Images</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="metric">
        <h2>42</h2>
        <p>Urdu Classes</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="metric">
        <h2>95.38%</h2>
        <p>Test Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="metric">
        <h2>ResNet18</h2>
        <p>Deep Learning Model</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; padding:25px; color:#a78bfa;">
    Built with PyTorch & Streamlit | Urdu AI Studio
</div>
""", unsafe_allow_html=True)
