import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Reshape, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam

# Step 1: Load the Dataset
def load_data(train_dir):
    images = []
    labels = []
    
    for filename in os.listdir(train_dir):
        img_path = os.path.join(train_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Unable to read image {img_path}. Skipping.")
            continue
        img = cv2.resize(img, (128, 128))
        images.append(img)
        label = filename.split(".")[0]
        labels.append(label)
    
    return np.array(images), np.array(labels)

# Step 2: Preprocess Data
def preprocess_data(images, labels):
    images = images.astype('float32') / 255.0
    images = np.stack([images] * 3, axis=-1)  # Convert grayscale to 3-channel by stacking
    
    char_list = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    max_plate_len = 8
    num_classes = len(char_list)
    
    def encode_label(label):
        one_hot_label = np.zeros((max_plate_len, num_classes), dtype='float32')
        for i, char in enumerate(label):
            if char in char_list:
                one_hot_label[i][char_list.index(char)] = 1
        return one_hot_label
    
    one_hot_labels = np.array([encode_label(label) for label in labels])
    return images, one_hot_labels

# Step 3: Build the Model using EfficientNetB0
def build_model():
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    
    # Unfreeze the last few layers of EfficientNet for fine-tuning
    for layer in base_model.layers[-20:]:
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
    
    model.compile(optimizer=Adam(learning_rate=0.0001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# Step 4: Learning Rate Scheduler
def lr_schedule(epoch, lr):
    if epoch < 10:
        return lr
    elif epoch < 20:
        return lr * 0.5
    else:
        return lr * 0.1

# Step 5: Train the Model
def train_model(model, X_train, y_train, X_val, y_val):
    # Advanced augmentation techniques using MixUp
    def mixup(images, labels, alpha=0.2):
        batch_size = images.shape[0]
        indices = np.random.permutation(batch_size)
        shuffled_images = images[indices]
        shuffled_labels = labels[indices]
        
        # Generate a lambda for each sample in the batch
        lam = np.random.beta(alpha, alpha, batch_size).astype('float32')
        lam_images = lam.reshape(batch_size, 1, 1, 1)
        lam_labels = lam.reshape(batch_size, 1, 1)
        
        # Perform MixUp
        mixed_images = lam_images * images + (1 - lam_images) * shuffled_images
        mixed_labels = lam_labels * labels + (1 - lam_labels) * shuffled_labels
        return mixed_images, mixed_labels

    # Custom data generator with MixUp
    def data_generator(X, y, batch_size):
        while True:
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                mixed_X, mixed_y = mixup(batch_X, batch_y)
                yield mixed_X, mixed_y

    # Initialize ImageDataGenerator for additional augmentations
    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    train_datagen.fit(X_train)
    
    train_generator = data_generator(X_train, y_train, batch_size=32)
    val_datagen = ImageDataGenerator()
    val_generator = val_datagen.flow(X_val, y_val, batch_size=32)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    
    model.fit(train_generator,
              steps_per_epoch=len(X_train) // 32,
              epochs=30,
              validation_data=val_generator,
              callbacks=[early_stopping, lr_scheduler])
    
    model.save('efficientnet_license_plate_model.h5')

# Step 6: Make Predictions and Generate Submission
def generate_submission_corrected(model, test_dir, submission_file):
    submission_template = pd.read_csv('submission_template.csv')
    
    test_images = []
    img_ids = []
    
    for filename in os.listdir(test_dir):
        img_path = os.path.join(test_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Unable to read image {img_path}. Skipping.")
            continue
        img = cv2.resize(img, (128, 128))
        img = np.stack([img] * 3, axis=-1)  # Convert grayscale to 3-channel by stacking
        test_images.append(img)
        img_ids.append(filename.split(".")[0])
    
    test_images = np.array(test_images).astype('float32') / 255.0
    
    print(f"Number of test images: {len(test_images)}")  # Debugging line
    
    predictions = model.predict(test_images)
    
    print(f"Shape of predictions: {predictions.shape}")  # Debugging line
    
    char_list = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    submission_rows = []
    
    for i, pred in enumerate(predictions):
        # pred has shape (8, 36)
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
    if len(images) == 0:
        raise ValueError("No images found in the training directory. Please check the path.")
    images, labels = preprocess_data(images, labels)
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    # Build and train the model
    model = build_model()
    train_model(model, X_train, y_train, X_val, y_val)
    
    # Generate submission
    generate_submission_corrected(model, test_dir, 'efficientnet_submission.csv')
