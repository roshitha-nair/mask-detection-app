import cv2
import numpy as np
import gradio as gr
from tensorflow.keras.models import load_model

# Load model and face detector
model = load_model("model.h5")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def detect_mask(image):
    try:
        # Convert to RGB (Gradio gives RGB already, but added safety)
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            return "❌ No face detected"

        for (x, y, w, h) in faces:
            face_img = rgb_img[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (224, 224))
            face_img = face_img / 255.0
            face_img = np.expand_dims(face_img, axis=0)

            prediction = model.predict(face_img)
            mask_prob = prediction[0][0]
            print(f"Prediction: {mask_prob}")

            return "😷 Wearing Mask" if mask_prob > 0.5 else "🛑 Not Wearing Mask"

    except Exception as e:
        print("Error during detection:", e)
        return f"⚠️ Error: {str(e)}"

# Gradio interface
iface = gr.Interface(
    fn=detect_mask,
    inputs=gr.Image(type="numpy", source="upload", tool="editor", label="Upload Image or Use Webcam"),
    outputs=gr.Text(label="Mask Detection Result"),
    title="Face Mask Detector",
    description="Upload a photo or use your webcam to check if a person is wearing a mask.",
    allow_flagging="never"
)

iface.launch()
