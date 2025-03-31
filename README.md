# Image Classification for Refund Items

This project provides an automated image classification system for refund items, focusing on classifying different shoe types (boots, flip-flops, sandals). 
The model is trained using a feature-based approach with Support Vector Machines and is designed for batch processing of new refund requests.

## Motivation and purpose
The project was developed for the course "Project: From Model to Production" (DLBDSMTP01) at the International University of Applied Sciences, tutored by Prof. Dr. Frank Passing. 

## Project Structure
```
├── app.py                     # Flask API to serve the model
├── image_classification_model.py  # Script to train the model
├── scaler.py                  # Script for feature scaling
├── run_batch.py               # Script for batch classification
├── model/                     # Directory for storing the trained model
├── dataset/                   # Directory for storing the dataset
├── new_products/              # Directory for storing new images for classification
├── results/                   # Directory to store classification results
├── requirements.txt           # Dependencies
├── .gitignore                 # Files to be ignored in Git
├── README.md                  # Project documentation
```

## Setup Instructions

### 1. Install Dependencies
Make sure you have Python installed, then install the required packages:
```sh
pip install -r requirements.txt
```

### 2. Download Model and Dataset
Since the model and dataset are too large for GitHub, please download them manually:
- **Trained Model:** [Download from Google Drive](https://drive.google.com/file/d/1ERvs3NAXEbBm8PB85okIfELZ3jr6U_J-/view?usp=sharing)
- **Dataset:** [Download from Google Drive](https://drive.google.com/drive/folders/19CJox1GWZLqGB4tyqU2LSVN1uCLbOkN1?usp=sharing)

After downloading, place them in the following directories:
```
model/image_classifier.pkl

# For training (if needed)
dataset/shoeTypeClassifierDataset/
```

### 3. Run the Model API
To start the Flask API for classification:
```sh
python app.py
```
The API will be accessible at `http://127.0.0.1:5000`.

### 4. Classify New Images in Batches
To classify new images stored in `new_products/`, run:
```sh
python run_batch.py
```
Results will be saved in `results/classification_results.json`.

## How It Works
- The **image classification model** extracts handcrafted features (histograms, edge detection, LBP, and HOG) and uses an SVM classifier.
- The **Flask API** serves the trained model, allowing images to be classified via HTTP requests.
- The **batch classification script** processes multiple images overnight via cron jobs or manual execution.



