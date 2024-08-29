import os
import numpy as np
import pandas as pd
import cv2

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Reshape, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Step 1: Load and Preprocess Data (same as before)
def load_data(train_dir):
    images = []
    labels = []
    
    for filename in os.listdir(train_dir):
        img_path = os.path.join(train_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))
        images.append(img)
        label = filename.split(".")[0]
        labels.append(label)
    
    return np.array(images), np.array(labels)

def preprocess_data(images, labels):
    images = images / 255.0
    images = np.stack([images] * 3, axis=-1)  # Convert grayscale to 3-channel by stacking
    
    char_list = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    max_plate_len = 8
    num_classes = len(char_list)
    
    def encode_label(label):
        one_hot_label = np.zeros((max_plate_len, num_classes))
        for i, char in enumerate(label):
            if char in char_list:
                one_hot_label[i][char_list.index(char)] = 1
        return one_hot_label
    
    one_hot_labels = np.array([encode_label(label) for label in labels])
    return images, one_hot_labels

# Step 2: Define the Model with EfficientNet
def build_model(learning_rate=0.001, dropout_rate=0.5):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(dropout_rate),
        Dense(128, activation='relu'),
        Dropout(dropout_rate),
        Dense(8 * 36, activation='softmax'),
        Reshape((8, 36))
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Step 3: Learning Rate Scheduler with Cyclical Learning Rate
def cyclical_learning_rate(step_size, min_lr, max_lr):
    def clr_schedule(epoch):
        cycle = np.floor(1 + epoch / (2 * step_size))
        x = np.abs(epoch / step_size - 2 * cycle + 1)
        lr = min_lr + (max_lr - min_lr) * np.maximum(0, (1 - x))
        return lr
    return clr_schedule

# Step 4: Train the Model
def train_model(model, X_train, y_train, X_val, y_val):
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    train_datagen.fit(X_train)
    
    # Validation data generator without augmentation
    val_datagen = ImageDataGenerator()
    
    # Cyclical Learning Rate
    clr = LearningRateScheduler(cyclical_learning_rate(step_size=10, min_lr=0.0001, max_lr=0.001))
    
    # Callbacks for learning rate scheduling and early stopping
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    
    # Fit the model
    model.fit(train_datagen.flow(X_train, y_train, batch_size=32),
              epochs=50,
              validation_data=val_datagen.flow(X_val, y_val),
              callbacks=[clr, lr_reduction, early_stopping])
    
    model.save('efficientnet_license_plate_model.h5')

# Step 5: Generate Submission (same as before)
def generate_submission_corrected(model, test_dir, submission_file):
    submission_template = pd.read_csv('submission_template.csv')
    
    test_images = []
    img_ids = []
    
    for filename in os.listdir(test_dir):
        img_path = os.path.join(test_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))
        img = np.stack([img] * 3, axis=-1)  # Convert grayscale to 3-channel by stacking
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
    # Load and preprocess data
    train_dir = 'data/train_data/final_train_set'
    test_dir = 'data/test_data/final_test_set'
    
    images, labels = load_data(train_dir)
    images, labels = preprocess_data(images, labels)
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    # Build and train the model
    model = build_model()
    train_model(model, X_train, y_train, X_val, y_val)
    
    # Generate submission
    generate_submission_corrected(model, test_dir, 'enhanced_submission.csv')
