import cv2
import numpy as np
import gradio as gr
from tensorflow.keras.models import load_model

# Load your trained mask detection model
model = load_model('model.h5')

# Load face detector (Haar cascade or any other)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_mask(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (224, 224))
        face_img = face_img / 255.0
        face_img = np.expand_dims(face_img, axis=0)

        prediction = model.predict(face_img)
        mask_prob = prediction[0][0]
        
        label = "Wearing Mask" if mask_prob > 0.5 else "Not Wearing Mask"
        color = (0, 255, 0) if mask_prob > 0.5 else (0, 0, 255)
        
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return image

iface = gr.Interface(
    fn=detect_mask,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Image(type="numpy"),
    title="Real-Time Face Mask Detection",
    description="Upload an image to detect whether people are wearing masks."
)

if __name__ == "__main__":
    iface.launch()
