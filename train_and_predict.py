import os
import numpy as np
import pandas as pd
import cv2
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Reshape
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from tensorflow.keras.applications import ResNet50, InceptionV3, Xception
from tensorflow.keras.optimizers import Adam

# Step 1: Load the Dataset
def load_data(train_dir, labels_csv):
    images = []
    labels = []
    
    # Load labels from CSV
    label_df = pd.read_csv(labels_csv)
    
    # Create a dictionary for quick lookup
    label_dict = {row['img_id']: row['plate_number'] for _, row in label_df.iterrows()}
    
    for filename in os.listdir(train_dir):
        img_id = filename.split(".")[0]
        img_path = os.path.join(train_dir, filename)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Skipping corrupted image: {filename}")
                continue
            img = cv2.resize(img, (128, 128))
            images.append(img)
            label = label_dict.get(img_id, None)
            if label is not None:
                labels.append(label)
            else:
                print(f"No label found for {filename}, skipping this image.")
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
            continue
    
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

# Step 3: Build and Train Multiple Models
def build_model(base_model):
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
    
    # Define class weights (example: more weight to less frequent characters)
    weights = np.ones((36,))  # Default weights are 1 for all classes
    
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=custom_weighted_categorical_crossentropy(weights), metrics=['accuracy'])
    return model

# Custom Weighted Loss Function
def custom_weighted_categorical_crossentropy(weights):
    def loss(y_true, y_pred):
        return tf.reduce_mean(tf.reduce_sum(-y_true * tf.math.log(y_pred) * weights, axis=-1))
    return loss

# Learning Rate Scheduler
def lr_schedule(epoch, lr):
    if epoch > 30:
        return lr * 0.1
    elif epoch > 20:
        return lr * 0.5
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)

# Train the Model
def train_model(model, X_train, y_train, X_val, y_val):
    # Enhanced training data generator with more augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=15,  # Increased rotation for more variation
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,  # Increased shear for more distortion
        zoom_range=0.15,   # Increased zoom for better generalization
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],  # Randomly adjust brightness
        channel_shift_range=30.0,  # Randomly shift the color channels
        fill_mode='nearest'  # Filling strategy for any new pixels created
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
    
    return model

# Step 4: Ensemble the Models
def ensemble_predictions(models, X_test):
    predictions = [model.predict(X_test) for model in models]
    averaged_predictions = np.mean(predictions, axis=0)
    return averaged_predictions

# Make Predictions and Generate Submission
def generate_submission_corrected(predictions, submission_file, template_path):
    submission_template = pd.read_csv(template_path)
    
    char_list = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    for i, pred in enumerate(predictions):
        pred = pred.reshape(8, 36)  # Reshape to (8, 36)
        for j, char_probs in enumerate(pred):
            pred_char = np.argmax(char_probs)
            row_index = (i * 8) + j
            submission_template.iloc[row_index, 1:] = 0  # Set all to 0
            submission_template.iloc[row_index, 1 + pred_char] = 1  # Set the correct character to 1
    
    submission_template.to_csv(submission_file, index=False)

if __name__ == '__main__':
    train_dir = 'data/train_data/final_train_set'
    test_dir = 'data/test_data/final_test_set'
    labels_csv = 'data/train_labels.csv' 
    template_path = 'data/submission_template.csv'
    
    # Load and preprocess data
    images, labels = load_data(train_dir, labels_csv)
    images, labels = preprocess_data(images, labels)
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    # Build and train multiple models
    resnet_model = build_model(ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3)))
    inception_model = build_model(InceptionV3(weights='imagenet', include_top=False, input_shape=(128, 128, 3)))
    xception_model = build_model(Xception(weights='imagenet', include_top=False, input_shape=(128, 128, 3)))
    
    trained_resnet_model = train_model(resnet_model, X_train, y_train, X_val, y_val)
    trained_inception_model = train_model(inception_model, X_train, y_train, X_val, y_val)
    trained_xception_model = train_model(xception_model, X_train, y_train, X_val, y_val)
    
    # Load and preprocess test data
    test_images, _ = load_data(test_dir, labels_csv)
    test_images = test_images / 255.0
    test_images = np.stack([test_images] * 3, axis=-1)
    
    # Ensemble the models
    ensemble_preds = ensemble_predictions([trained_resnet_model, trained_inception_model, trained_xception_model], test_images)
    
    # Generate submission
    generate_submission_corrected(ensemble_preds, 'ensemble_submission.csv', template_path)
