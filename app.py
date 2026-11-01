
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import requests
from io import BytesIO

st.set_page_config(
    page_title="Urdu AI Studio",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- DESIGN ----------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #101018;
    color: #f5f3ff;
    font-family: 'Inter', sans-serif;
}

header {visibility: hidden;}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 32px;
    border: 1px solid #302a48;
    border-radius: 20px;
    background: linear-gradient(120deg, #211a36, #171724);
    margin-bottom: 28px;
}

.brand {
    color: #b69aff;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 3px;
}

.hero h1 {
    color: #ffffff;
    font-size: 36px;
    font-weight: 800;
    margin: 15px 0 8px 0;
}

.hero p {
    color: #aaa6bb;
    font-size: 15px;
}

.tag {
    display: inline-block;
    background: #29213e;
    color: #c8b5ff;
    padding: 7px 13px;
    border-radius: 30px;
    font-size: 12px;
    border: 1px solid #51416d;
}

.section-heading {
    color: #ffffff;
    font-size: 19px;
    font-weight: 700;
    margin-bottom: 15px;
}

div[data-testid="stFileUploader"] {
    background: #191923;
    border: 1px dashed #69518f;
    border-radius: 16px;
    padding: 18px;
}

div[data-testid="stFileUploader"] label,
div[data-testid="stFileUploader"] p {
    color: #ded9ed !important;
}

.stButton > button {
    background: #8056d9;
    color: white;
    border: 1px solid #a17cf0;
    border-radius: 11px;
    min-height: 48px;
    font-weight: 700;
    width: 100%;
    transition: 0.2s;
}

.stButton > button:hover {
    background: #956fe9;
    border-color: #c4aaff;
    color: white;
}

div[data-testid="stImage"] img {
    border-radius: 14px;
}

.result-box {
    background: linear-gradient(145deg, #241d37, #191923);
    border: 1px solid #55416f;
    border-radius: 18px;
    padding: 25px;
    text-align: center;
    margin-top: 10px;
}

.result-label {
    color: #aaa1bd;
    font-size: 13px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.urdu-character {
    color: #d0b8ff;
    font-size: 100px;
    font-weight: 700;
    line-height: 1.5;
    margin: 8px;
}

.metric-card {
    background: #1b1b28;
    border: 1px solid #343044;
    border-radius: 14px;
    padding: 18px;
    margin-top: 15px;
}

.metric-title {
    color: #a9a3b8;
    font-size: 12px;
}

.metric-value {
    color: #ffffff;
    font-size: 25px;
    font-weight: 700;
    margin-top: 5px;
}

.info-card {
    background: #191923;
    border: 1px solid #302d40;
    border-radius: 16px;
    padding: 20px;
    margin-top: 25px;
}

.info-card h4 {
    color: #c7adff;
}

.info-card p {
    color: #aaa6b8;
    font-size: 13px;
}

.footer {
    border-top: 1px solid #302d40;
    margin-top: 40px;
    padding-top: 20px;
    color: #827d92;
    font-size: 12px;
    text-align: center;
}

.stProgress > div > div > div > div {
    background: #a17cf0;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.markdown("""
<div class="hero">
    <div class="brand">✦ URDU AI STUDIO</div>
    <h1>Handwritten Character Recognition</h1>
    <p>Transform handwritten Urdu characters into intelligent predictions using Deep Learning.</p>
    <span class="tag">RESNET18 MODEL</span>
    &nbsp;
    <span class="tag">42 CHARACTER CLASSES</span>
</div>
""", unsafe_allow_html=True)

# ---------------- MODEL ----------------

MODEL_URL = "https://huggingface.co/sameera123-2026/urdu-handwritten-character-recognition/resolve/main/model.pth"

LABELS_URL = "https://huggingface.co/sameera123-2026/urdu-handwritten-character-recognition/resolve/main/labels.json"


@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 42)

    response = requests.get(MODEL_URL, timeout=120)
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
    response = requests.get(LABELS_URL, timeout=30)
    response.raise_for_status()
    return json.loads(response.text)


# ---------------- TRANSFORM ----------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- MAIN APPLICATION ----------------

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown(
        '<div class="section-heading">01 &nbsp; Upload Image</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose your handwritten character",
        type=["png", "jpg", "jpeg"],
        help="Upload one isolated Urdu character."
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded handwritten character",
            width=300
        )

        st.success("Image ready for recognition.")

    recognize = st.button(
        "✦  Analyze Character",
        use_container_width=True
    )

with right:
    st.markdown(
        '<div class="section-heading">02 &nbsp; AI Prediction</div>',
        unsafe_allow_html=True
    )

    if uploaded_file is None:
        st.markdown("""
        <div class="result-box">
            <div class="result-label">PREDICTION AREA</div>
            <div style="font-size:65px; padding:25px;">✧</div>
            <p style="color:#aaa6bb;">
                Your predicted Urdu character will appear here.
            </p>
        </div>
        """, unsafe_allow_html=True)

    elif recognize:
        try:
            with st.spinner("Analyzing your character..."):

                model = load_model()
                labels = load_labels()

                image_tensor = transform(image).unsqueeze(0)

                with torch.no_grad():
                    output = model(image_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    confidence, prediction = torch.max(
                        probabilities, dim=1
                    )

                character = labels[prediction.item()]
                score = confidence.item() * 100

            st.markdown(f"""
            <div class="result-box">
                <div class="result-label">PREDICTED CHARACTER</div>
                <div class="urdu-character" dir="rtl">{character}</div>
                <div style="color:#b9a8df;">
                    Recognition completed successfully
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-title">MODEL CONFIDENCE</div>'
                f'<div class="metric-value">{score:.2f}%</div>',
                unsafe_allow_html=True
            )
            st.progress(score / 100)
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as error:
            st.error(f"Prediction error: {error}")

    else:
        st.info("Click Analyze Character to start prediction.")

# ---------------- PROJECT INFORMATION ----------------

st.markdown("""
<div class="info-card">
    <h4>About This Project</h4>
    <p>
        This application recognizes isolated Urdu handwritten
        characters using a fine-tuned ResNet18 deep learning model.
    </p>
    <p>
        <b>Dataset:</b> Cursive-Text Mendeley v3
        &nbsp; | &nbsp;
        <b>Images:</b> 19,901
        &nbsp; | &nbsp;
        <b>Classes:</b> 42
    </p>
    <p>
        <b>Technologies:</b> Python, PyTorch, Hugging Face and Streamlit
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    URDU AI STUDIO &nbsp; • &nbsp;
    DEEP LEARNING PROJECT &nbsp; • &nbsp;
    POWERED BY RESNET18
    <br><br>
    <a href="https://github.com/sameera566/urdu-handwritten-character-recognition"
       style="color:#b69aff;text-decoration:none;">
       View GitHub Repository
    </a>
</div>
""", unsafe_allow_html=True)
