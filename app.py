import cv2
import numpy as np
import streamlit as st
from keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from collections import deque

model = load_model("model.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

CONFIDENCE_THRESHOLD = 0.85
HYSTERESIS_MARGIN = 0.1
SMOOTHING_FRAMES = 30
predictions_queue = deque(maxlen=SMOOTHING_FRAMES)

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_label = None
        self.last_confidence = 0.0

    def preprocess_face(self, face_img):
        face = cv2.resize(face_img, (150, 150))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face / 255.0
        return np.expand_dims(face, axis=0)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_img = img[y:y+h, x:x+w]

            if face_img.size != 0:
                face_input = self.preprocess_face(face_img)
                confidence = model.predict(face_input, verbose=0)[0][0]
                predictions_queue.append(confidence)

                avg_confidence = sum(predictions_queue) / len(predictions_queue)

                if self.last_label is None:
                    label = 1 if avg_confidence >= CONFIDENCE_THRESHOLD else 0
                else:
                    if self.last_label == 1 and avg_confidence < CONFIDENCE_THRESHOLD - HYSTERESIS_MARGIN:
                        label = 0
                    elif self.last_label == 0 and avg_confidence > CONFIDENCE_THRESHOLD + HYSTERESIS_MARGIN:
                        label = 1
                    else:
                        label = self.last_label

                self.last_label = label
                self.last_confidence = avg_confidence

                label_text = "Wearing Mask" if label == 0 else "Not Wearing Mask"
                color = (0, 255, 0) if label == 0 else (0, 0, 255)
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return img

st.title("Real-Time Face Mask Detection")
webrtc_streamer(key="mask-detection", video_processor_factory=VideoProcessor, rtc_configuration=RTC_CONFIGURATION)
