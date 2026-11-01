import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import requests
from io import BytesIO

st.set_page_config(
    page_title="Urdu AI Recognition",
    page_icon="📝",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top left, #25204b, #090b18 55%);
    color: #ffffff;
}

header {visibility: hidden;}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
}

.hero {
    text-align: center;
    padding: 35px 20px;
    background: linear-gradient(135deg, #211b42, #15152e);
    border: 1px solid #51418c;
    border-radius: 22px;
    box-shadow: 0 0 35px rgba(124, 58, 237, 0.18);
    margin-bottom: 30px;
}

.hero h1 {
    color: white;
    font-size: 38px;
    margin-bottom: 10px;
}

.hero p {
    color: #c4b5fd;
    font-size: 17px;
}

.section-title {
    color: #c4b5fd;
    font-size: 22px;
    font-weight: bold;
    margin-bottom: 18px;
}

.panel {
    background: rgba(27, 28, 53, 0.9);
    border: 1px solid #40376c;
    border-radius: 20px;
    padding: 25px;
    min-height: 350px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
}

.stButton > button {
    background: linear-gradient(90deg, #7c3aed, #a855f7);
    color: white;
    border: none;
    border-radius: 12px;
    height: 48px;
    font-weight: bold;
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #9333ea, #c084fc);
    color: white;
    border: 1px solid #c4b5fd;
}

div[data-testid="stFileUploader"] {
    background: #17172e;
    border: 1px dashed #8b5cf6;
    border-radius: 14px;
    padding: 15px;
}

div[data-testid="stFileUploader"] label,
.stMarkdown p,
.stMarkdown h3 {
    color: #eeeeff;
}

.prediction {
    background: linear-gradient(135deg, #30215c, #211a43);
    border: 1px solid #8b5cf6;
    border-radius: 20px;
    padding: 25px;
    text-align: center;
    font-size: 90px;
    color: white;
    margin: 20px 0;
    box-shadow: 0 0 25px rgba(139,92,246,0.2);
}

.footer {
    text-align: center;
    color: #a5a0c5;
    padding: 25px;
    margin-top: 30px;
    border-top: 1px solid #393456;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>📝 Urdu Handwritten Character Recognition</h1>
    <p>Powered by Deep Learning and ResNet18</p>
</div>
""", unsafe_allow_html=True)

MODEL_URL = "https://huggingface.co/sameera123-2026/urdu-handwritten-character-recognition/resolve/main/model.pth"
LABELS_URL = "https://huggingface.co/sameera123-2026/urdu-handwritten-character-recognition/resolve/main/labels.json"

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 42)

    response = requests.get(MODEL_URL)
    response.raise_for_status()

    state_dict = torch.load(
        BytesIO(response.content),
        map_location="cpu"
    )

    model.load_state_dict(state_dict)
    model.eval()
    return model

@st.cache_data
def load_labels():
    response = requests.get(LABELS_URL)
    response.raise_for_status()
    return json.loads(response.text)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

left, right = st.columns(2, gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📤 Upload Your Character</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose an Urdu character image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Your Uploaded Image", width=250)
        st.success("Image uploaded successfully!")

    recognize = st.button("✨ Recognize Character")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🤖 AI Prediction Result</div>', unsafe_allow_html=True)

    if uploaded_file and recognize:
        try:
            with st.spinner("AI is analyzing your character..."):
                model = load_model()
                labels = load_labels()

                image_tensor = transform(image).unsqueeze(0)

                with torch.no_grad():
                    output = model(image_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    confidence, prediction = torch.max(probabilities, dim=1)

                character = labels[prediction.item()]
                score = confidence.item() * 100

            st.success("Recognition Complete!")

            st.markdown(
                f'<div class="prediction" dir="rtl">{character}</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<h3 style="text-align:center;color:#c4b5fd;">'
                f'Predicted Character</h3>',
                unsafe_allow_html=True
            )

            st.metric("Model Confidence", f"{score:.2f}%")

        except Exception as e:
            st.error(f"Error: {e}")

    elif not uploaded_file:
        st.info("Upload an image to see the AI prediction.")

    else:
        st.info("Click Recognize Character to start prediction.")

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    <b>Model:</b> ResNet18 &nbsp; | &nbsp;
    <b>Dataset:</b> 19,901 Images &nbsp; | &nbsp;
    <b>Classes:</b> 42
    <br><br>
    Built with PyTorch, Hugging Face and Streamlit
    <br><br>
    © Urdu Handwritten Character Recognition Project
</div>
""", unsafe_allow_html=True)
