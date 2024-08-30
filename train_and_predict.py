import os
import numpy as np
import pandas as pd
import cv2

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape, BatchNormalization, Dropout, GlobalAveragePooling2D, Input, Bidirectional, LSTM
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split

# Step 1: Load the Dataset
def load_data(train_dir):
    images = []
    labels = []
    
    for filename in os.listdir(train_dir):
        img_path = os.path.join(train_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))
        
        # Apply adaptive thresholding
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        images.append(img)
        label = filename.split(".")[0]
        labels.append(label)
    
    return np.array(images), np.array(labels)

# Step 2: Preprocess Data
def preprocess_data(images, labels):
    images = images / 255.0
    images = np.stack([images] * 3, axis=-1)  # Convert grayscale to 3-channel by stacking
    
    char_list = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    max_plate_len = 8  # Assuming license plates have a maximum of 8 characters
    num_classes = len(char_list) + 1  # +1 for the blank label used in CTC
    
    def encode_label(label):
        encoded = []
        for char in label:
            if char in char_list:
                encoded.append(char_list.index(char) + 1)  # Shift by 1 to leave 0 as the blank label
            else:
                print(f"Warning: Character '{char}' in label '{label}' not found in char_list, skipping.")
        
        # Padding with 0, which is reserved for CTC blank label
        while len(encoded) < max_plate_len:
            encoded.append(0)
        
        # Truncate the label if it's longer than max_plate_len
        if len(encoded) > max_plate_len:
            encoded = encoded[:max_plate_len]
        
        return encoded
    
    encoded_labels = np.array([encode_label(label) for label in labels], dtype=np.int32)
    
    return images, encoded_labels

# Step 3: Build the CRNN Model
def build_crnn_model(input_shape=(128, 128, 3), num_classes=37, max_text_len=8):
    input_img = Input(shape=input_shape, name='image_input')
    
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_img)
    base_model.trainable = True
    
    # Unfreeze the last few layers of ResNet50
    for layer in base_model.layers[:-10]:
        layer.trainable = False
    
    x = base_model.output
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)  # Add a Conv2D layer to adjust dimensions
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)  # Adjust the pooling to maintain more features
    
    # Flatten before reshaping
    x = Flatten()(x)
    x = Dense(max_text_len * 256, activation='relu')(x)  # Adjust to ensure the correct number of elements
    x = Reshape((max_text_len, 256))(x)  # Reshape to (max_text_len, 256)
    
    # Add Bidirectional LSTM for sequence modeling
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_img, outputs=x)
    
    # Define CTC loss function with dynamic shape handling
    def ctc_loss(y_true, y_pred):
        batch_size = K.shape(y_pred)[0]
        input_length = K.shape(y_pred)[1]
        label_length = K.shape(y_true)[1]

        input_length = tf.fill([batch_size, 1], input_length)
        label_length = tf.fill([batch_size, 1], label_length)

        return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    
    model.compile(optimizer='adam', loss=ctc_loss, metrics=['accuracy'])
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
    # Training data generator with augmentation including perspective transformations
    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        preprocessing_function=lambda img: cv2.warpAffine(img, 
                                                         cv2.getRotationMatrix2D((64, 64), 10 * np.random.randn(), 1), 
                                                         (128, 128))
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
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        img = np.stack([img] * 3, axis=-1)  # Convert grayscale to 3-channel by stacking
        test_images.append(img)
        img_ids.append(filename.split(".")[0])
    
    test_images = np.array(test_images) / 255.0
    
    predictions = model.predict(test_images)
    
    char_list = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    submission_rows = []
    
    for i, pred in enumerate(predictions):
        pred = pred.reshape(8, 37)  # Reshape to (8, 37)
        for j, char_probs in enumerate(pred):
            pred_char = np.argmax(char_probs)
            one_hot_vector = [0] * 36  # 36 classes for one-hot encoding excluding the blank label
            if pred_char > 0:  # Ignore blank label (0)
                one_hot_vector[pred_char - 1] = 1  # Shift index back by 1 to fit in one-hot vector
            submission_row = [f"{img_ids[i]}_{j + 1}"] + one_hot_vector
            submission_rows.append(submission_row)
    
    # Create the DataFrame with the same columns as the submission template
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
    model = build_crnn_model(input_shape=(128, 128, 3), num_classes=37, max_text_len=8)
    train_model(model, X_train, y_train, X_val, y_val)
    
    # Generate submission
    generate_submission_corrected(model, test_dir, 'enhanced_submission.csv')
