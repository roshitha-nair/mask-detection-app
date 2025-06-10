from tensorflow.keras.models import load_model
import cv2
import numpy as np
import gradio as gr

# Load your trained model
model = load_model("model.h5")

# Define class labels
labels = ["Mask", "No Mask"]

# Preprocessing and prediction function
def predict_mask(image):
    img = cv2.resize(image, (224, 224))  # Adjust based on your model input
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)[0]
    label = labels[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    return f"{label} ({confidence:.2f})"

# Gradio interface
iface = gr.Interface(fn=predict_mask, inputs="image", outputs="text", title="Face Mask Detection")
iface.launch()
