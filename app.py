import cv2
import numpy as np
import gradio as gr
from tensorflow.keras.models import load_model

# Load model and face detector
model = load_model("model.h5")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def detect_mask(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return "No face detected"

    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (224, 224))
        face_img = face_img / 255.0
        face_img = np.expand_dims(face_img, axis=0)
        prediction = model.predict(face_img)
        mask_prob = prediction[0][0]
        return "Wearing Mask" if mask_prob > 0.5 else "Not Wearing Mask"

# Gradio interface with image input and label output
iface = gr.Interface(
    fn=detect_mask,
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs=gr.Text(label="Mask Status"),
    title="Face Mask Detector",
    description="Upload a photo to check if the person is wearing a mask"
)

iface.launch()
