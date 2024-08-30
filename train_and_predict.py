import os
import numpy as np
import pandas as pd
import cv2

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape, BatchNormalization, Dropout, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler # type: ignore
from tensorflow.keras.applications import ResNet50 # type: ignore
from sklearn.model_selection import train_test_split

# Step 1: Load the Dataset
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

# Step 2: Preprocess Data
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

# Step 3: Build the Model using Transfer Learning
def build_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    
    # Unfreeze the top layers of ResNet50
    for layer in base_model.layers[-10:]:
        layer.trainable = True
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(8 * 36, activation='softmax'),
        Reshape((8, 36))
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Step 4: Learning Rate Scheduler
def lr_schedule(epoch, lr):
    if epoch > 30:
        return lr * 0.1
    elif epoch > 20:
        return lr * 0.5
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)

# Step 5: Train the Model
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
    
    # Creating iterators for training and validation
    train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=32)
    
    # Callbacks for reducing learning rate and early stopping
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    
    # Fit the model
    model.fit(train_generator,
              epochs=50,
              validation_data=val_generator,
              callbacks=[lr_scheduler, lr_reduction, early_stopping])
    
    # Save the model
    model.save('enhanced_license_plate_model.h5')

# Step 6: Make Predictions and Generate Submission
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
    
    print(f"Number of test images: {len(test_images)}")  # Debugging line
    
    predictions = model.predict(test_images)
    
    print(f"Shape of predictions: {predictions.shape}")  # Debugging line
    
    char_list = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    submission_rows = []
    
    for i, pred in enumerate(predictions):
        print(f"Prediction shape before reshape: {pred.shape}")  # Debugging line
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
    train_dir = 'data/train_data/final_train_set'
    test_dir = 'data/test_data/final_test_set'
    
    # Load and preprocess data
    images, labels = load_data(train_dir)
    images, labels = preprocess_data(images, labels)
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    # Build and train the model
    model = build_model()
    train_model(model, X_train, y_train, X_val, y_val)
    
    # Generate submission
    generate_submission_corrected(model, test_dir, 'enhanced_submission.csv')
