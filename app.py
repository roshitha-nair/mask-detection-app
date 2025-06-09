import cv2
import numpy as np
import gradio as gr
from tensorflow.keras.models import load_model

model = load_model('model.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_mask(frame):
    if frame is None:
        return "No frame provided"
    
    # frame from gr.Camera is RGB, convert to BGR for OpenCV
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
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
        if mask_prob > 0.5:
            return "Mask detected"
    return "No mask detected"

iface = gr.Interface(
    fn=detect_mask,
    inputs=gr.Camera(),
    outputs=gr.Textbox(),
    title="Face Mask Detector",
    description="Detect if a face in the webcam frame is wearing a mask."
)

iface.launch()
