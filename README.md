# Urdu Handwritten Character Recognition

## Project Overview
This project uses Deep Learning to recognize isolated Urdu handwritten characters from images.

The system takes an image of a handwritten Urdu character and predicts which Urdu character it represents.

## Dataset
- **Dataset:** Cursive-Text Mendeley v3
- **Total Images:** 19,901
- **Number of Classes:** 42
- **Source:** https://data.mendeley.com/datasets/k5fz57zd9z/3

## Pretrained Model
- **Model:** ResNet18
- **Source:** Hugging Face
- **Pretrained Model:** https://huggingface.co/timm/resnet18.a1_in1k

## Training Results
The model was fine-tuned for 5 epochs.

- **Validation Accuracy:** 95.13%
- **Test Accuracy:** 95.38%

## Technologies Used
- Python
- PyTorch
- Torchvision
- ResNet18
- Hugging Face
- Streamlit

## Hugging Face Model
Fine-tuned model and labels:

https://huggingface.co/sameera123-2026/urdu-handwritten-character-recognition

## Live Demo
Try the application here:

https://urdu-handwritten-character-recognition-6fz9kxj2gfou88iyxxgtvu.streamlit.app/

## How to Run Locally

1. Clone this repository:

   `git clone https://github.com/sameera566/urdu-handwritten-character-recognition.git`

2. Open the project folder:

   `cd urdu-handwritten-character-recognition`

3. Install the required libraries:

   `pip install -r requirements.txt`

4. Run the Streamlit application:

   `streamlit run app.py`

## Project Goal
The goal of this project is to apply transfer learning to recognize Urdu handwritten characters and make the trained model available through a web application.
