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
from sklearn.model_selection import train_test_split, GridSearchCV

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

# Step 3: Define the model function
def build_model(learning_rate=0.001, dropout_rate=0.5, base_trainable=False):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = base_trainable  # Set base model to trainable or not

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

# Step 4: Manual Grid Search Implementation
def manual_grid_search(X_train, y_train, X_val, y_val):
    best_score = 0
    best_params = None
    param_grid = {
        'learning_rate': [0.001, 0.0001],
        'dropout_rate': [0.5, 0.3],
        'batch_size': [32, 64],
        'base_trainable': [False, True]
    }
    
    for lr in param_grid['learning_rate']:
        for dr in param_grid['dropout_rate']:
            for bs in param_grid['batch_size']:
                for bt in param_grid['base_trainable']:
                    print(f"Testing model with lr={lr}, dropout_rate={dr}, batch_size={bs}, base_trainable={bt}")
                    model = build_model(learning_rate=lr, dropout_rate=dr, base_trainable=bt)
                    model.fit(X_train, y_train, epochs=10, batch_size=bs, validation_data=(X_val, y_val), verbose=0)
                    score = model.evaluate(X_val, y_val, verbose=0)[1]  # Get accuracy
                    print(f"Validation accuracy: {score}")
                    if score > best_score:
                        best_score = score
                        best_params = {'learning_rate': lr, 'dropout_rate': dr, 'batch_size': bs, 'base_trainable': bt}
    
    print(f"Best score: {best_score}")
    print(f"Best parameters: {best_params}")
    
    return best_params

# Step 5: Retrain the model with the best parameters
def retrain_best_model(X_train, y_train, best_params):
    model = build_model(learning_rate=best_params['learning_rate'], 
                        dropout_rate=best_params['dropout_rate'], 
                        base_trainable=best_params['base_trainable'])
    model.fit(X_train, y_train, epochs=10, batch_size=best_params['batch_size'], validation_split=0.2)
    return model

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
    # Load and preprocess data
    train_dir = 'data/train_data/final_train_set'
    test_dir = 'data/test_data/final_test_set'
    
    images, labels = load_data(train_dir)
    images, labels = preprocess_data(images, labels)
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    # Perform manual grid search
    best_params = manual_grid_search(X_train, y_train, X_val, y_val)
    
    # Retrain the best model
    best_model = retrain_best_model(X_train, y_train, best_params)
    
    # Generate submission with the best model
    generate_submission_corrected(best_model, test_dir, 'enhanced_submission.csv')
