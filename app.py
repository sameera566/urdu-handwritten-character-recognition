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
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #312e81, #581c87);
    color: white;
}
h1, h2, h3, p, label {
    color: white !important;
}
.main-title {
    text-align: center;
    font-size: 36px;
    font-weight: bold;
    color: white;
    padding: 15px;
}
.subtitle {
    text-align: center;
    color: #e0e7ff;
    font-size: 17px;
}
.stButton > button {
    background: linear-gradient(90deg, #7c3aed, #db2777);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 25px;
    font-weight: bold;
    width: 100%;
}
.stButton > button:hover {
    background: #6d28d9;
    color: white;
}
.result-card {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 18px;
    padding: 25px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">📝 Urdu Handwritten Character Recognition</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Upload an Urdu handwritten character and let AI recognize it.</div>',
    unsafe_allow_html=True
)

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

st.markdown("### 📤 Upload Your Character")
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Character", width=250)

    if st.button("✨ Recognize Character"):
        with st.spinner("AI is recognizing your character..."):
            try:
                model = load_model()
                labels = load_labels()

                image_tensor = transform(image).unsqueeze(0)

                with torch.no_grad():
                    output = model(image_tensor)
                    prediction = torch.argmax(output, dim=1).item()

                st.success("Recognition Complete!")

                st.markdown(
                    '<div class="result-card">'
                    '<h3>Predicted Urdu Character</h3>'
                    f'<h1 style="font-size:75px;">{labels[prediction]}</h1>'
                    '</div>',
                    unsafe_allow_html=True
                )

            except Exception as e:
                st.error(f"Error: {e}")
