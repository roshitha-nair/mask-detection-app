import cv2
import numpy as np
import gradio as gr
from tensorflow.keras.models import load_model

model = load_model('model.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_mask_label(image):
    if image is None:
        return "No image provided"
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
        if mask_prob > 0.5:
            return "Mask detected"
    return "No mask detected"

def detect_from_webcam(frame):
    # frame is BGR numpy array from webcam
    return detect_mask_label(frame)

with gr.Blocks() as demo:
    mode = gr.Radio(choices=["Upload Image", "Webcam"], label="Select Input Mode")

    upload_image = gr.Image(type="numpy", label="Upload Image")
    webcam_image = gr.Image(source="webcam", streaming=True, type="numpy", label="Webcam")

    output_text = gr.Textbox(label="Detection Result")

    def process(input_mode, upload_img, webcam_img):
        if input_mode == "Upload Image":
            return detect_mask_label(upload_img)
        else:
            return detect_mask_label(webcam_img)

    mode.change(fn=lambda x: None, inputs=mode, outputs=[upload_image, webcam_image])
    
    submit_btn = gr.Button("Detect Mask")
    submit_btn.click(fn=process, inputs=[mode, upload_image, webcam_image], outputs=output_text)

demo.launch()
