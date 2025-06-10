import gradio as gr
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

# Load trained model
model = load_model('model.h5')

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Preprocessing function
def preprocess_face(face_img):
    face = cv2.resize(face_img, (150, 150))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face / 255.0
    return np.expand_dims(face, axis=0)

# Prediction function
def detect_mask(image):
    img = np.array(image.convert("RGB"))
    img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "❌ No face detected"

    (x, y, w, h) = faces[0]  # Use the first detected face
    face_img = img_cv[y:y+h, x:x+w]

    if face_img.size == 0 or w == 0 or h == 0:
        return "⚠️ Invalid face region"

    face_input = preprocess_face(face_img)
    result = model.predict(face_input, verbose=0)
    confidence = result[0][0]

    if confidence < 0.5:
        return f"✅ Wearing Mask (Confidence: {1 - confidence:.2f})"
    else:
        return f"❌ Not Wearing Mask (Confidence: {confidence:.2f})"

# Gradio Interface
iface = gr.Interface(
    fn=detect_mask,
    inputs=gr.Image(type="pil", label="Upload or Capture Image"),
    outputs="text",
    title="Face Mask Detection",
    description="Upload an image or use webcam to detect if a person is wearing a face mask.",
    allow_flagging="never"
)

iface.launch(share=True)
