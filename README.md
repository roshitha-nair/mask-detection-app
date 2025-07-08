# Real-Time Face Mask Detection using Deep Learning and Gradio

This project is a real-time face mask detection system built using a Convolutional Neural Network (CNN) model trained with Keras/TensorFlow.  
It uses OpenCV for face detection and Gradio for a simple web interface.  
Users can upload an image or capture a photo using their webcam to check if a person is wearing a face mask.

---

## 🚀 Features

- 🤖 CNN-based binary classification (**Wearing Mask / Not Wearing Mask**)
- 🧠 Face detection using Haar Cascade
- 📸 Image upload and webcam capture (via Gradio)
- 📈 Confidence score displayed with result
- 🌐 Gradio public URL for easy access and sharing

---

## 📌 How to Use

1. Click **"Upload or Capture Image"**
2. Upload an image or take a webcam snapshot
3. The app will display one of the following results:
   - ✅ **Wearing Mask** (Confidence: `0.xx`)
   - ❌ **Not Wearing Mask** (Confidence: `0.xx`)
   - ⚠️ **No Face Detected** (if no face found)

---

## 📦 Deployment Status

- ✅ Local testing using Gradio completed  
- ⏳ Deployment to Hugging Face Spaces is **in progress**

Once testing is finalized, the app will be deployed publicly on Hugging Face Spaces for broader access.

---
