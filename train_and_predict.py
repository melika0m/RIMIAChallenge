
import os
import numpy as np
import pandas as pd
import cv2

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape # type: ignore
from sklearn.model_selection import train_test_split

# Step 1: Load the Dataset
def load_data(train_dir):
    images = []
    labels = []
    
    # Assuming train_dir contains folders where images are stored with their corresponding plate number as filenames
    for filename in os.listdir(train_dir):
        # Load each image
        img_path = os.path.join(train_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
        img = cv2.resize(img, (128, 128))  # Resize for uniformity
        images.append(img)
        
        # Extract label (plate number) from the filename
        label = filename.split(".")[0]
        labels.append(label)
    
    return np.array(images), np.array(labels)

# Step 2: Preprocess Data
def preprocess_data(images, labels):
    # Normalize images
    images = images / 255.0
    
    # Convert labels to one-hot encoding (simplified for the demonstration)
    char_list = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    max_plate_len = 8  # Assuming max plate length is 8 characters
    num_classes = len(char_list)
    
    def encode_label(label):
        one_hot_label = np.zeros((max_plate_len, num_classes))
        for i, char in enumerate(label):
            if char not in char_list:
                print(f"Unexpected character '{char}' in label '{label}'")
            else:
                one_hot_label[i][char_list.index(char)] = 1
        return one_hot_label
    
    one_hot_labels = np.array([encode_label(label) for label in labels])
    
    return images, one_hot_labels

# Step 3: Build the Model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(8 * 36, activation='softmax'),
        Reshape((8, 36))  # Add this layer to reshape the output to (8, 36)
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Step 4: Train the Model
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    model.save('license_plate_model.h5')

# Step 5: Make Predictions and Generate Submission
def generate_submission_corrected(model, test_dir, submission_file):
    # Load the submission template to get the column names
    submission_template = pd.read_csv('submission_template.csv')
    
    test_images = []
    img_ids = []
    
    for filename in os.listdir(test_dir):
        img_path = os.path.join(test_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))
        test_images.append(img)
        img_ids.append(filename.split(".")[0])
    
    test_images = np.array(test_images) / 255.0
    predictions = model.predict(test_images)
    
    char_list = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    submission_rows = []
    
    for i, pred in enumerate(predictions):
        pred = pred.reshape(8, 36)  # Reshape to (8, 36)
        for j, char_probs in enumerate(pred):
            pred_char = np.argmax(char_probs)
            one_hot_vector = [0] * 36
            one_hot_vector[pred_char] = 1
            submission_row = [f"{img_ids[i]}_{j + 1}"] + one_hot_vector
            submission_rows.append(submission_row)
    
    submission_df = pd.DataFrame(submission_rows, columns=submission_template.columns)
    submission_df.to_csv(submission_file, index=False)


if __name__ == '__main__':
    # Directories
    train_dir = 'data/train_data/final_train_set'
    test_dir = 'data/test_data/final_test_set'
    
    # Load and preprocess data
    images, labels = load_data(train_dir)
    images, labels = preprocess_data(images, labels)
    
    # Reshape images to add channel dimension
    images = images.reshape(-1, 128, 128, 1)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    # Build and train model
    model = build_model()
    train_model(model, X_train, y_train)
    
    # Generate submission
    generate_submission_corrected(model, test_dir, 'submission.csv')
