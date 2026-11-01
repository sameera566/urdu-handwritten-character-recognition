
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

# ---------- DESIGN ----------
st.markdown("""
<style>
.stApp {
    background: #080b1c;
    color: white;
}
header {visibility:hidden;}
.block-container {max-width:1250px; padding-top:2rem;}
h1,h2,h3,p,label {color:white;}
.hero {
    padding:28px;
    border:1px solid #493078;
    border-radius:20px;
    background:linear-gradient(120deg,#21104a,#10152e);
    margin-bottom:25px;
}
.hero h1 {font-size:40px;font-weight:800;}
.hero span {color:#bd8aff;}
.panel {
    background:#12152e;
    border:1px solid #39305e;
    border-radius:18px;
    padding:22px;
    margin-bottom:15px;
}
.result {
    background:linear-gradient(145deg,#29145a,#12152e);
    border:1px solid #9355dc;
    border-radius:18px;
    padding:25px;
    text-align:center;
}
.big-urdu {
    font-size:100px;
    color:white;
    font-family:serif;
}
.metric {
    background:#171b39;
    border:1px solid #493078;
    border-radius:14px;
    padding:18px;
    text-align:center;
}
.metric h2 {color:#c49aff;}
.stButton button {
    background:linear-gradient(90deg,#9333ea,#5b45d6);
    color:white;
    border:0;
    border-radius:12px;
    height:45px;
    font-weight:700;
}
.stButton button:hover {border:1px solid #d8b4fe;}
.stFileUploader section {
    background:#171633;
    border:1px dashed #9b67dc;
    border-radius:14px;
}
</style>
""", unsafe_allow_html=True)

# ---------- MODEL ----------
@st.cache_resource
def load_model():
    r = requests.get(MODEL_URL, timeout=120)
    r.raise_for_status()

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 42)
    model.load_state_dict(
        torch.load(BytesIO(r.content), map_location="cpu")
    )
    model.eval()
    return model

@st.cache_data
def load_labels():
    r = requests.get(LABELS_URL, timeout=60)
    r.raise_for_status()
    return json.loads(r.text)

def recognize(image):
    model = load_model()
    labels = load_labels()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    tensor = transform(image.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
        scores, indices = torch.topk(probs, 3)

    results = []
    for score, index in zip(scores, indices):
        results.append({
            "character": labels[index.item()],
            "confidence": score.item() * 100
        })

    return results

# ---------- SESSION STATE ----------
if "history" not in st.session_state:
    st.session_state.history = []

if "result" not in st.session_state:
    st.session_state.result = None

if "image" not in st.session_state:
    st.session_state.image = None

# ---------- HEADER ----------
st.markdown("""
<div class="hero">
    <h1>🧠 Urdu <span>AI Studio</span></h1>
    <p>Intelligent Urdu Handwritten Character Recognition</p>
    <p>Upload • Recognize • Explore Results</p>
</div>
""", unsafe_allow_html=True)

# ---------- NAVIGATION ----------
tab_home, tab_history, tab_info = st.tabs([
    "🏠 Recognition",
    "📜 History",
    "ℹ️ Model Information"
])

# ---------- RECOGNITION ----------
with tab_home:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("📤 Upload Character")
        st.write("Upload one handwritten Urdu character image.")

        uploaded = st.file_uploader(
            "Choose image",
            type=["png", "jpg", "jpeg"],
            key="character_upload"
        )

        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.session_state.image = image
            st.image(image, caption="Image Preview", width=250)

        col1, col2 = st.columns(2)

        with col1:
            predict_btn = st.button(
                "✨ Predict",
                use_container_width=True,
                disabled=st.session_state.image is None
            )

        with col2:
            reset_btn = st.button(
                "↻ Reset",
                use_container_width=True
            )

        if reset_btn:
            st.session_state.result = None
            st.session_state.image = None
            st.session_state.history = []
            st.rerun()

        if predict_btn:
            try:
                with st.spinner("AI is analyzing your character..."):
                    results = recognize(st.session_state.image)

                st.session_state.result = results
                st.session_state.history.insert(0, results)

            except Exception as e:
                st.error(f"Prediction error: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("🤖 AI Prediction")

        if st.session_state.result:
            results = st.session_state.result
            best = results[0]

            st.markdown(f"""
            <div class="result">
                <p>Best Predicted Character</p>
                <div class="big-urdu">{best['character']}</div>
                <h2>{best['confidence']:.2f}%</h2>
                <p>Model confidence score</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### Top 3 Predictions")

            for i, item in enumerate(results):
                st.write(
                    f"**{i+1}. {item['character']}** — "
                    f"{item['confidence']:.2f}%"
                )
                st.progress(
                    min(1.0, max(0.0, item["confidence"] / 100))
                )

        else:
            st.info("Upload an image and click Predict to see results.")

        st.markdown('</div>', unsafe_allow_html=True)

# ---------- HISTORY ----------
with tab_history:
    st.subheader("📜 Prediction History")

    if st.session_state.history:
        for number, record in enumerate(st.session_state.history, 1):
            best = record[0]
            with st.expander(
                f"Prediction {number} — {best['character']}"
            ):
                for item in record:
                    st.write(
                        f"{item['character']} : "
                        f"{item['confidence']:.2f}%"
                    )
    else:
        st.info("No predictions yet. Recognize an image first.")

    if st.session_state.history:
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

# ---------- MODEL INFORMATION ----------
with tab_info:
    st.subheader("ℹ️ Model Information")

    a, b, c = st.columns(3)

    with a:
        st.markdown("""
        <div class="metric">
        <h2>ResNet18</h2>
        <p>Pretrained Architecture</p>
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown("""
        <div class="metric">
        <h2>42</h2>
        <p>Character Classes</p>
        </div>
        """, unsafe_allow_html=True)

    with c:
        st.markdown("""
        <div class="metric">
        <h2>95.38%</h2>
        <p>Test Accuracy</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Model Repository")
    st.link_button(
        "Open Hugging Face Model",
        "https://huggingface.co/sameera123-2026/urdu-handwritten-character-recognition"
    )

st.divider()
st.caption("Urdu AI Studio | Built with PyTorch and Streamlit")
