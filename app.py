import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import tensorflow as tf
from PIL import Image
import base64
import time

# Load the model
model = tf.keras.models.load_model("model.h5")

# Define preprocessing function 

def preprocess(image):
    image_color = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_color,(32,32))
    image_for_prediction  = np.array([image_resized]).astype(np.float32)/255.0
    print(image_for_prediction.shape)
    return image_for_prediction


# Define prediction function
def predict(hand_image_path,model):
    image = preprocess(hand_image_path)
    predicted_letter = [np.argmax(model.predict(image),axis =1 )]
    return predicted_letter


# Streamlit UI
st.title("ASL Sign Language Recognition")
st.markdown("Capture hand sign every 5 seconds and classify using your model.")

# Create a placeholder for video frame
frame_placeholder = st.empty()
prediction_placeholder = st.empty()

# JavaScript + HTML to show webcam
st.markdown(
    """
    <style>
    #webcam-container {
        text-align: center;
    }
    video {
        border: 3px solid #00bfff;
        border-radius: 12px;
    }
    </style>
    <div id="webcam-container">
        <video autoplay playsinline muted id="webcam" width="480" height="360"></video>
    </div>
    <script>
    const video = document.getElementById('webcam');
    async function setupCamera() {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
    }
    setupCamera();
    </script>
    """,
    unsafe_allow_html=True
)

# Use OpenCV + Streamlit WebRTC/Frames
run = st.checkbox('Start Prediction')

if run:
    # Use OpenCV to access camera
    cap = cv2.VideoCapture(0)
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame")
            break

        # Define bounding box (hardcoded for simplicity)
        h, w, _ = frame.shape
        box_start = (int(w / 2 - 100), int(h / 2 - 100))
        box_end = (int(w / 2 + 100), int(h / 2 + 100))
        cv2.rectangle(frame, box_start, box_end, (0, 255, 0), 2)

        # Show frame in Streamlit
        frame_placeholder.image(frame, channels="BGR")

        # Every 5 seconds, capture and predict
        if time.time() - last_time >= 5:
            crop = frame[box_start[1]:box_end[1], box_start[0]:box_end[0]]
            class_idx, confidence = predict(crop)
            prediction_placeholder.markdown(f"### Prediction: {class_idx} (Confidence: {confidence:.2f})")
            last_time = time.time()

    cap.release()


