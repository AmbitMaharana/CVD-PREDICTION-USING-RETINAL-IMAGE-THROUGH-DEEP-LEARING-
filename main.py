# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from google.colab import files
import os

# Step 1: Upload CSV and Images
print("Please upload the CSV file containing image IDs and labels:")
uploaded = files.upload()

csv_file = list(uploaded.keys())[0]
data_path = '/content/drive/MyDrive/Evaluation_Set/RFMiD_Validation_Labels.csv'
data = pd.read_csv(data_path)
print("Uploaded CSV:")
print(data.head())

# Upload images
print("Please upload image files:")
uploaded_images = files.upload()

# Save images to local directory
image_dir = "/content/drive/MyDrive/Evaluation_Set/Validation"
os.makedirs(image_dir, exist_ok=True)
for name, content in uploaded_images.items():
    with open(os.path.join(image_dir, name), "wb") as f:
        f.write(content)

# Step 2: Define image paths and labels
images = data['ID'].apply(lambda x: os.path.join(image_dir, f"{x}.png"))
labels = data['Disease_Risk']  # Adjust column name based on your CSV

# Step 3: Split data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

# Image preprocessing function
def preprocess_image(image_path):
    """Load and preprocess an image."""
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    return img_array

# Preprocess all images
train_images = np.array([preprocess_image(img) for img in train_images])
val_images = np.array([preprocess_image(img) for img in val_images])
train_labels = np.array(train_labels)
val_labels = np.array(val_labels)

# Step 4: Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 5: Train the model
history = model.fit(
    train_images, train_labels,
    validation_data=(val_images, val_labels),
    epochs=10,
    batch_size=32
)

# Evaluate the model
val_preds = (model.predict(val_images) > 0.5).astype(int)
print("Classification Report:")
print(classification_report(val_labels, val_preds))

# Step 6: Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.show()

# Step 7: Predict on new images
print("Upload new images for prediction:")
new_images = files.upload()

for name, content in new_images.items():
    with open(os.path.join(image_dir, name), "wb") as f:
        f.write(content)

new_image_paths = [os.path.join(image_dir, name) for name in new_images.keys()]
new_images_preprocessed = np.array([preprocess_image(img) for img in new_image_paths])

predictions = (model.predict(new_images_preprocessed) > 0.5).astype(int)
for img_path, pred in zip(new_image_paths, predictions):
    print(f"Image: {img_path}, Predicted Risk: {'High' if pred == 1 else 'Low'}")
