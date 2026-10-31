
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import requests
from io import BytesIO

st.set_page_config(
    page_title="Urdu Handwritten Character Recognition",
    page_icon="📝"
)

st.title("Urdu Handwritten Character Recognition")
st.write("Upload an Urdu handwritten character image to recognize it.")

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

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Character", width=250)

    if st.button("Recognize Character"):
        with st.spinner("Recognizing..."):
            try:
                model = load_model()
                labels = load_labels()

                image_tensor = transform(image).unsqueeze(0)

                with torch.no_grad():
                    output = model(image_tensor)
                    prediction = torch.argmax(output, dim=1).item()

                st.success("Recognition Complete!")
                st.subheader("Predicted Urdu Character")
                st.title(str(labels[prediction]))

            except Exception as e:
                st.error(f"Error: {e}")
