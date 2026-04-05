# Facial Keypoints Detection

## Overview
This project contains my deliverable for the Facial Keypoints Detection competition. It primarily focuses on detecting facial keypoints from grayscale face images using a convolutional neural network. The goal of this competition is to predict the (x,y) coordinates of key facial features like eyes, nose, and mouth.

You can find the Kaggle competition here: https://www.kaggle.com/competitions/facial-keypoints-detection/

## Project Structure

```
Facial_Keypoints_Detection/
│
├── src/
│   ├── config.py             
│   ├── dataset.py            
│   ├── make_submission.py    
│   ├── model.py              
│   ├── predict.py            
│   └── train.py              
├── submissions/
│   ├── submission-1.csv      
│   ├── submission-2.csv      
│   └── submission-3.csv      
├── requirements.txt          
└── README.md
```
## Approach

### 1. Data Preprocessing
- Loaded 96x96 grayscale images
- Normalized pixel values
- Handled missing keypoint values
- converted data into PyTorch tensors

### 2. Model Architecture
- Convolutional Neural Network (CNN)
- Multiple convolution + pooling layers
- Fully connected layers for regression output
- Output layer predicts keypoint coordinates

For this task, I treated the problem as a regression problem where the model directly predicts landmark coordinates.

### Training
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam
- Trained over 30 epochs
- Evaluated using validation loss

### Results
- Successfully trained the CNN to predict facial keypoints
- Improved performance from first submission to the last submission
- Top 20 public scores on the competition leaderboard

## Files included
- config.py
   - contains main parameters and constants used in this project
- dataset.py
   - loads the dataset downloaded from the competition page, preprocesses the images and labels, and applies augmentation techniques
- make_submission.py
   - converts predictions made by model into a kaggle-ready submission format
- model.py
   - defines neural network architecure
- predict.py
   - loads the trained model and makes predictions on the test dataset
- train.py
   - loads the dataset, trains the model, and saves the best model weights
- requirements.txt
   - dependencies needed to run the project

## How to Run
From the root Directory:
- pip install -r requirements.txt

Train the model:
- cd src
- python .\train.py

While in the src directory, make predictions:
- python .\predict.py