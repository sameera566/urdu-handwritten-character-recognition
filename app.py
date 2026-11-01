
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

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at 80% 10%, #21105a 0%, transparent 35%),
        radial-gradient(ellipse at 5% 50%, #170c3c 0%, transparent 35%),
        #07091b;
    font-family: 'Inter', sans-serif;
    color: #ffffff;
}

header, footer {visibility: hidden;}

.block-container {
    max-width: 1400px;
    padding: 2rem 3rem 2rem 3rem;
}

h1, h2, h3, p, label {
    color: #ffffff !important;
}

.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 15px 22px;
    border: 1px solid #30215d;
    border-radius: 18px;
    background: rgba(17, 15, 45, 0.88);
    margin-bottom: 45px;
}

.brand {
    font-size: 23px;
    font-weight: 800;
    color: white;
}

.brand span {color: #a855f7;}

.nav-right {
    color: #c4b5fd;
    font-size: 14px;
    font-weight: 600;
}

.hero {
    padding: 10px 5px 35px 5px;
}

.hero h1 {
    font-size: clamp(32px, 4vw, 52px);
    font-weight: 800;
    line-height: 1.15;
    margin-bottom: 12px;
    letter-spacing: -1.5px;
}

.gradient-text {
    background: linear-gradient(90deg, #c084fc, #7c3aed, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #aaa8d0 !important;
    font-size: 17px;
}

.section-card {
    background: linear-gradient(145deg, #171438, #0d102b);
    border: 1px solid #493078;
    border-radius: 22px;
    padding: 26px;
    min-height: 390px;
    box-shadow: 0 15px 45px rgba(0,0,0,0.25);
}

.card-heading {
    font-size: 22px;
    font-weight: 750;
    margin-bottom: 6px;
}

.card-description {
    color: #aaa8d0 !important;
    font-size: 14px;
    margin-bottom: 22px;
}

.upload-symbol {
    font-size: 42px;
    color: #c084fc;
    text-align: center;
    padding: 15px;
}

.result-box {
    background:
        radial-gradient(circle at 50% 0%, #392071, transparent 65%),
        #100d2c;
    border: 1px solid #8546e9;
    border-radius: 20px;
    padding: 25px;
    text-align: center;
    box-shadow: 0 0 30px rgba(124,58,237,0.15);
}

.result-character {
    font-size: 105px;
    color: #ffffff;
    line-height: 1.5;
    font-family: serif;
}

.result-label {
    color: #c4b5fd;
    font-size: 13px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.score {
    font-size: 34px;
    font-weight: 800;
    color: #c084fc;
    margin-top: 10px;
}

.stFileUploader section {
    background: #11102c !important;
    border: 1px dashed #9661e9 !important;
    border-radius: 15px !important;
}

.stFileUploader section:hover {
    border-color: #c084fc !important;
}

.stButton button {
    background: linear-gradient(100deg, #9333ea, #5b35df) !important;
    border: 1px solid #a855f7 !important;
    color: white !important;
    border-radius: 13px !important;
    height: 52px !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    box-shadow: 0 5px 25px rgba(124,58,237,0.3);
    transition: 0.2s ease;
}

.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(168,85,247,0.5);
}

.stats-title {
    font-size: 23px;
    font-weight: 750;
    margin: 30px 0 15px 0;
}

.stat-card {
    background: linear-gradient(145deg, #19143b, #10132d);
    border: 1px solid #38265f;
    border-radius: 17px;
    padding: 20px 12px;
    text-align: center;
}

.stat-number {
    font-size: 25px;
    font-weight: 800;
    color: #c084fc;
}

.stat-name {
    color: #aaa8d0;
    font-size: 13px;
    margin-top: 5px;
}

.footer {
    text-align: center;
    color: #77749f;
    border-top: 1px solid #292044;
    margin-top: 40px;
    padding: 22px;
    font-size: 13px;
}

div[data-testid="stImage"] img {
    border: 2px solid #7041b8;
    border-radius: 15px;
}

.stAlert {
    background: #171438;
    border: 1px solid #493078;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- MODEL ----------------

@st.cache_resource
def load_model():
    response = requests.get(MODEL_URL, timeout=120)
    response.raise_for_status()

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 42)

    state = torch.load(
        BytesIO(response.content),
        map_location="cpu"
    )
    model.load_state_dict(state)
    model.eval()

    return model


@st.cache_data
def load_labels():
    response = requests.get(LABELS_URL, timeout=60)
    response.raise_for_status()
    return json.loads(response.text)


def predict_character(image, model, labels):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    tensor = transform(image.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)
        score, index = torch.max(probabilities, dim=1)

    return labels[index.item()], score.item() * 100


# ---------------- HEADER ----------------

st.markdown("""
<div class="navbar">
    <div class="brand">🧠 Urdu <span>AI Studio</span></div>
    <div class="nav-right">✦ SMART CHARACTER RECOGNITION</div>
</div>

<div class="hero">
    <h1>Urdu Handwritten<br>
    Character <span class="gradient-text">Recognition</span></h1>
    <p>Transform handwritten Urdu characters into AI predictions.</p>
</div>
""", unsafe_allow_html=True)


# ---------------- MAIN PANELS ----------------

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("""
    <div class="section-card">
        <div class="card-heading">📤 Upload Your Character</div>
        <div class="card-description">
            Select an image containing one handwritten Urdu character.
        </div>
        <div class="upload-symbol">⇧</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Your uploaded character", width=170)

    recognize = st.button(
        "✦  Recognize Character",
        use_container_width=True,
        disabled=uploaded_file is None
    )

with right:
    st.markdown("""
    <div class="section-card">
        <div class="card-heading">🤖 Prediction Result</div>
        <div class="card-description">
            Your recognition result will appear below.
        </div>
    """, unsafe_allow_html=True)

    if uploaded_file and recognize:
        try:
            with st.spinner("Loading AI model and recognizing..."):
                model = load_model()
                labels = load_labels()
                character, confidence = predict_character(
                    image, model, labels
                )

            st.markdown(f"""
            <div class="result-box">
                <div class="result-label">Predicted Character</div>
                <div class="result-character">{character}</div>
                <div class="result-label">Recognition complete</div>
                <div class="score">{confidence:.2f}%</div>
                <div class="result-label">Model confidence score</div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as error:
            st.error(f"Prediction failed: {error}")

    else:
        st.markdown("""
        <div class="result-box">
            <div class="result-label">Waiting for image</div>
            <div class="result-character">؟</div>
            <div class="result-label">
                Upload an image to see the prediction
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------- STATISTICS ----------------

st.markdown(
    '<div class="stats-title">✦ Model Overview</div>',
    unsafe_allow_html=True
)

a, b, c, d = st.columns(4)

stats = [
    (a, "19,901", "Dataset Images"),
    (b, "42", "Urdu Classes"),
    (c, "95.38%", "Test Accuracy"),
    (d, "ResNet18", "AI Architecture")
]

for col, number, name in stats:
    with col:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{number}</div>
            <div class="stat-name">{name}</div>
        </div>
        """, unsafe_allow_html=True)


# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">
    Built with ❤️ using PyTorch and Streamlit
    <br><br>
    Urdu AI Studio | Handwritten Character Recognition
</div>
""", unsafe_allow_html=True)
