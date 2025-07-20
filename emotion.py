import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image

st.title("Face Emotion Recognition App")
st.write("Upload an image or take a snapshot with your webcam to detect emotions on faces.")

# --- Image Upload ---
st.header("Detect from Uploaded Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# --- Webcam Input ---
st.header("Detect from Webcam Snapshot")
img_file_buffer = st.camera_input("Take a picture")

def process_image(image):
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    if len(faces) == 0:
        st.warning("No faces detected.")
        st.image(img_array, caption="No faces detected", use_column_width=True)
    else:
        for i, (x, y, w, h) in enumerate(faces):
            face_roi = img_array[y:y + h, x:x + w]
            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                emotion_scores = result[0]['emotion']
            except Exception as e:
                emotion = "Error"
                emotion_scores = {}
            cv2.rectangle(img_array, (x, y), (x + w, y + h), (0,0,255), 2)
            cv2.putText(img_array, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            st.write(f"Face {i+1} at ({x},{y}):")
            st.json(emotion_scores)
        st.image(img_array, caption="Detected Emotions", use_column_width=True)

# Process uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    process_image(image)

# Process webcam image
if img_file_buffer is not None:
    image = Image.open(img_file_buffer).convert('RGB')
    process_image(image)
