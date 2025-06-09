import cv2
import numpy as np
from keras.models import load_model

# Load model
model = load_model('model.h5')

# Load image from your path
img_path = r"C:\roshitha\Own projects\Face Mask Detection\sample_withoutmask.jpeg"
img = cv2.imread(img_path)

# Preprocess image same as training
img_resized = cv2.resize(img, (150, 150))
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
img_normalized = img_rgb / 255.0
img_reshaped = np.reshape(img_normalized, (1, 150, 150, 3))

# Predict
result = model.predict(img_reshaped)
label = 1 if result[0][0] > 0.5 else 0
labels_dict = {0: "Mask", 1: "NO Mask"}

print(f"Prediction: {labels_dict[label]}")
import matplotlib.pyplot as plt
plt.imshow(img_rgb)  # or img_normalized if you want normalized image
plt.title("Input image to model")
plt.show()
