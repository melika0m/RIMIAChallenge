License Plate Recognition with Ensemble of Deep Learning Models
This repository contains an end-to-end pipeline for recognizing license plate numbers using an ensemble of deep learning models (ResNet50, InceptionV3, and Xception). The model is trained with K-Fold cross-validation and utilizes data augmentation and a custom weighted loss function to improve accuracy.

Table of Contents
Overview
Installation
Data Preparation
Training
Inference
Submission
References
Overview
This project implements an ensemble of models to recognize license plates using a series of deep learning models with K-Fold cross-validation. The core components include:

Models: ResNet50, InceptionV3, and Xception (from TensorFlow/Keras)
K-Fold Cross Validation: Ensures robust training across multiple folds.
Ensemble Learning: Averages predictions from the models to enhance overall accuracy.
YOLOv5 (optional): Can be used for license plate detection.
Installation
Requirements
Python 3.12.2 or later
TensorFlow
OpenCV
NumPy
Pandas
Scikit-learn
Steps
Clone the repository:

bash
git clone <repository-url>
cd RimIA
Set up a virtual environment:
python -m venv rim-ai-env
.\rim-ai-env\Scripts\activate  # On Windows
source rim-ai-env/bin/activate  # On Linux or MacOS
Install dependencies:


pip install numpy pandas tensorflow opencv-python matplotlib scikit-learn pillow
Verify installation:


python --version
pip list  # To verify that all dependencies are installed
Data Preparation
Before training the model, ensure that your dataset is organized as follows:

bash
Copier le code
dataset/
│
├── images/
│   ├── train/
│   └── test/
│
├──
│ train_labels.csv
│
├── submission_template.csv
Train images: Place the training images inside images/train/.
Test images: Place the test images inside images/test/.
Labels: Ensure that the training labels are stored in labels/train_labels.csv with two columns:
img_id: Unique identifier for each image.
plate_number: The corresponding license plate number.
Submission template: Ensure the submission template is available at submission_template.csv.
Training
Cross-Validation & Model Training
The provided script trains an ensemble of models using 5-fold cross-validation:

Run the training script:


python train_and_predict.py
The script performs the following steps:

Loads the training data.
Preprocesses the data (resizing, normalization, and one-hot encoding).
Trains multiple models using cross-validation.
Saves the trained models.
Performs inference on test images and generates a submission file.
Key Parameters:
Image Size: 128x128 pixels (adjustable within the script).
K-Fold Cross-Validation: 5 splits.
Ensemble Models: ResNet50, InceptionV3, and Xception.
Inference
After training the model, the test images are passed through the ensemble of models for prediction.

Run the inference pipeline (part of train_and_predict.py):
The ensemble of models averages the predictions across multiple models.
Results are written to ensemble_submission.csv.
Submission
The output from the inference step will be saved in a CSV file (ensemble_submission.csv). This file will follow the required submission format:

Each row corresponds to one character in the license plate.
The format is compatible with the competition rules (8 characters for each license plate).
References
TensorFlow Documentation
OpenCV Documentation

Instructions Summary:
Install the dependencies.
Prepare the dataset in the correct format.
Run the training script to train and validate the model using K-fold cross-validation.
Generate the ensemble predictions on the test data.
Submit the results from the ensemble_submission.csv.
This project is designed for automatic license plate recognition using state-of-the-art deep learning models.
