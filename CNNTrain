import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Disable TensorFlow CPU feature guard message

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_sample_images
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load sample images (two images of different sizes)
data = load_sample_images().images

# Convert images to numpy array and normalize pixel values
X = np.array(data)
X = X / 255.0  # Normalize pixel values to [0, 1]

# Create sample labels
y = np.array([0, 1])

# Convert labels to one-hot encoded format
y = to_categorical(y, num_classes=2)

# Create a basic CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(427, 640, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model's architecture
model.summary()

# Train the model
history = model.fit(X, y, epochs=5, batch_size=1)

# Visualize training history
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')

plt.tight_layout()
plt.show()
