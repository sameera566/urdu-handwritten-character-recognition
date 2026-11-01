import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import requests
from io import BytesIO

st.set_page_config(
    page_title="Urdu Character Recognition",
    page_icon="📝",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: #faf9ff;
    color: #171044;
}

header {visibility: hidden;}

.main-title {
    background: linear-gradient(110deg, #e9ddff, #c8b5ff);
    color: #171044;
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    font-size: 34px;
    font-weight: 800;
    margin-bottom: 25px;
    border: 1px solid #c7b5ff;
}

.subtitle {
    text-align: center;
    color: #40336f;
    font-size: 17px;
    margin-bottom: 25px;
}

div[data-testid="stFileUploader"] {
    background: white;
    padding: 20px;
    border: 2px dashed #b9a5f5;
    border-radius: 15px;
}

.stButton > button {
    background: linear-gradient(90deg, #5b2dbb, #7545d5);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 13px;
    font-weight: bold;
    width: 100%;
}

.stButton > button:hover {
    background: #48219d;
    color: white;
}

.panel {
    background: white;
    border: 1px solid #ded4ff;
    border-radius: 16px;
    padding: 22px;
    min-height: 300px;
    box-shadow: 0 4px 15px rgba(70, 40, 130, 0.07);
}

.panel-title {
    background: #5429ad;
    color: white;
    padding: 15px;
    border-radius: 10px;
    font-size: 23px;
    font-weight: bold;
    margin-bottom: 20px;
}

.prediction {
    background: #eee7ff;
    color: #21104f;
    border-radius: 18px;
    padding: 25px;
    text-align: center;
    font-size: 75px;
    margin: 15px auto;
}

.footer {
    background: #f0ebff;
    border: 1px solid #ded4ff;
    border-radius: 15px;
    padding: 22px;
    text-align: center;
    color: #29145e;
    margin-top: 25px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-title">
📝 Urdu Handwritten Character Recognition
</div>
<div class="subtitle">
Upload an image of a handwritten Urdu character and get instant prediction.
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
    st.markdown(
        '<div class="panel-title">☁️ Upload Image</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose a handwritten Urdu character image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Character", width=250)
        st.success("Image uploaded successfully!")

    recognize = st.button("🔍 Recognize Character")

with right:
    st.markdown(
        '<div class="panel-title">✨ Prediction Result</div>',
        unsafe_allow_html=True
    )

    if uploaded_file and recognize:
        try:
            with st.spinner("AI is recognizing your character..."):
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

            st.success("Recognition Complete!")

            st.markdown(
                f'<div style="text-align:center;color:#171044;'
                f'font-size:24px;font-weight:bold;">'
                f'Predicted Character</div>'
                f'<div class="prediction" dir="rtl">{character}</div>',
                unsafe_allow_html=True
            )

            st.success(f"Confidence: {score:.2f}%")

        except Exception as e:
            st.error(f"Error: {e}")

    elif not uploaded_file:
        st.info("Upload an image to see the prediction here.")
    else:
        st.info("Click Recognize Character to get your result.")

st.markdown("""
<div class="footer">
<b>Dataset:</b> Cursive-Text Mendeley v3 (19,901 images, 42 classes)
<br><br>
<b>Model:</b> ResNet18 (Fine-tuned)
<br><br>
<b>Built With:</b> PyTorch • Streamlit
<br><br>
<a href="https://github.com/sameera566/urdu-handwritten-character-recognition">
View Project on GitHub
</a>
</div>
""", unsafe_allow_html=True)
