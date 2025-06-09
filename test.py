import cv2
import numpy as np
from keras.models import load_model
from collections import deque

model = load_model('model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

SMOOTHING_FRAMES = 30
CONFIDENCE_THRESHOLD = 0.85
HYSTERESIS_MARGIN = 0.1

predictions_queue = deque(maxlen=SMOOTHING_FRAMES)
last_label = None
last_confidence = 0.0
tracker = None
tracking = False
tracking_lost_counter = 0
TRACKING_LOST_THRESHOLD = 15
frames_since_detection = 0
DETECTION_INTERVAL = 15  # detect face every 15 frames
frames_since_last_pred = 0
PREDICTION_INTERVAL = 5  # predict every 5 frames

# Resize scale for faster detection & tracking
FRAME_SCALE = 0.5

cap = cv2.VideoCapture(0)

def scale_bbox(bbox, scale):
    x, y, w, h = bbox
    return (int(x/scale), int(y/scale), int(w/scale), int(h/scale))

def preprocess_face(face_img):
    face = cv2.resize(face_img, (150, 150))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face / 255.0
    return np.expand_dims(face, axis=0)

frames_since_last_detection = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0,0), fx=FRAME_SCALE, fy=FRAME_SCALE)
    gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    display_frame = frame.copy()

    if not tracking or frames_since_detection >= DETECTION_INTERVAL or tracking_lost_counter > TRACKING_LOST_THRESHOLD:
        faces = face_cascade.detectMultiScale(gray_small, 1.3, 5)
        frames_since_detection = 0
        tracking_lost_counter = 0

        if len(faces) > 0:
            (x_s, y_s, w_s, h_s) = faces[0]
            # Scale bbox back to original frame size
            x, y, w, h = scale_bbox((x_s, y_s, w_s, h_s), FRAME_SCALE)

            # Use a faster tracker: KCF or MOSSE
            tracker = cv2.TrackerKCF_create()
            tracker.init(frame, (x, y, w, h))
            tracking = True
            frames_since_last_detection = 0  # reset when face detected
        else:
            tracking = False
            predictions_queue.clear()
            last_label = None
            last_confidence = 0.0
            frames_since_last_detection += 1  # increase timer when no face

    else:
        tracking, bbox = tracker.update(frame)
        if tracking:
            x, y, w, h = tuple(map(int, bbox))
            tracking_lost_counter = 0
        else:
            tracking_lost_counter += 1

    if tracking:
        frames_since_last_pred += 1
        if frames_since_last_pred >= PREDICTION_INTERVAL:
            frames_since_last_pred = 0

            face_img = frame[y:y+h, x:x+w]

            # Skip empty or invalid face regions
            if face_img.size == 0 or w == 0 or h == 0:
                tracking = False
                predictions_queue.clear()
                last_label = None
                last_confidence = 0.0
            else:
                face_input = preprocess_face(face_img)
                result = model.predict(face_input, verbose=0)
                confidence = result[0][0]

                predictions_queue.append(confidence)
                avg_confidence = sum(predictions_queue) / len(predictions_queue)

                if last_label is None:
                    label = 1 if avg_confidence >= CONFIDENCE_THRESHOLD else 0
                else:
                    if last_label == 1 and avg_confidence < CONFIDENCE_THRESHOLD - HYSTERESIS_MARGIN:
                        label = 0
                    elif last_label == 0 and avg_confidence > CONFIDENCE_THRESHOLD + HYSTERESIS_MARGIN:
                        label = 1
                    else:
                        label = last_label

                last_label = label
                last_confidence = avg_confidence
                frames_since_last_detection = 0 # reset on confident prediction

        if last_label is not None:
            color = (0, 255, 0) if last_label == 0 else (0, 0, 255)
            label_text = 'Wearing Mask' if last_label == 0 else 'Not Wearing Mask'
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(display_frame, label_text, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    else:
        # Keep drawing last bbox briefly after tracking lost
        if (tracking_lost_counter < TRACKING_LOST_THRESHOLD or frames_since_last_detection < 30) and last_label is not None:
            color = (0, 255, 0) if last_label == 0 else (0, 0, 255)
            label_text = 'Wearing Mask' if last_label == 0 else 'Not Wearing Mask'
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(display_frame, label_text, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            frames_since_last_detection += 1  # increase timer while showing last box

        else:
            predictions_queue.clear()
            last_label = None
            last_confidence = 0.0

    frames_since_detection += 1

    cv2.imshow("Mask Detection Optimized", display_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
