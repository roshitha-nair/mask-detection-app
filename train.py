import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D,MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2

# Dataset directories
train_dir = 'dataset/train'
test_dir = 'dataset/test'

# Image Data Generator with augmentation for training
train_datagen = ImageDataGenerator(rescale=1./255,#normalizes pixel values to between 0 and 1.
                                   rotation_range=20,#randomly rotates images up to 20 degrees.
                                   zoom_range=0.2,#randomly zooms images up to 20%.
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
#Creates a data generator for testing images without augmentation, just normalization.
#We don't augment test data because we want to evaluate the model on original images.

# Load training and testing data
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),#resizes all images to 150x150 pixels.
                                                    batch_size=32,# processes 32 images at a time during training.
                                                    class_mode='binary',# 2 classes (mask or no mask).
                                                    shuffle=True)#means images are randomly shuffled every epoch to improve training.

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(150, 150),
                                                  batch_size=32,
                                                  class_mode='binary')

# Load MobileNetV2 base model, exclude top layers
base_model = MobileNetV2(input_shape=(150, 150, 3),
                         include_top=False,
                         weights='imagenet')

# Freeze the base model layers so they are not trained initially
base_model.trainable = False

# Build your model on top
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification output
])


# Compile Model
model.compile(optimizer=Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train Model
model.fit(train_generator,
          epochs=5,
          validation_data=test_generator)

# Optionally, unfreeze some layers for fine-tuning
base_model.trainable = True
# Freeze all layers except last few layers for fine-tuning
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Re-compile with a lower learning rate
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])
# Continue training for fine-tuning
model.fit(train_generator,
          epochs=5,
          validation_data=test_generator)

# Save Model
model.save('model.h5')
print("Model trained and saved as 'model.h5'")
