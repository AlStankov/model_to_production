from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
from skimage.feature import local_binary_pattern, hog
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib  # For saving and loading the model
from PIL import Image

# Initialize the Flask application
app = Flask(__name__)

# Define the path for saving uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load the pre-trained SVM model and scaler
svm = joblib.load('image_classifier.pkl')  # Path to the image classification model
scaler = joblib.load('scaler.pkl')  # Path to the scaler

# Classes in the same order as when you trained the model
classes = ['boots', 'flip_flops', 'sandals']

# Function to extract features from an image (same as your model)
def extract_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))  # Resize for consistency

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Color histogram
    hist_b = cv2.calcHist([image], [0], None, [8], [0, 256]).flatten()
    hist_g = cv2.calcHist([image], [1], None, [8], [0, 256]).flatten()
    hist_r = cv2.calcHist([image], [2], None, [8], [0, 256]).flatten()
    hist = np.hstack([hist_b, hist_g, hist_r])

    # Edge detection
    edges = cv2.Canny(gray, 100, 200)
    edge_hist = np.histogram(edges, bins=8, range=(0, 256))[0]

    # Texture analysis using LBP
    lbp = local_binary_pattern(gray, P=16, R=2, method='uniform')
    (lbp_hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 20), range=(0, 19))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)  # Normalize

    # HOG features
    hog_features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)

    # Combine features
    features = np.hstack([hist, edge_hist, lbp_hist, hog_features])
    return features

# Helper function to check allowed extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Extract features from the uploaded image
        features = extract_features(filepath)

        # Normalize the features
        features = scaler.transform([features])

        # Make prediction
        prediction = svm.predict(features)
        predicted_label = prediction[0]  # The predicted label is directly a string ("boots", "flip_flops", "sandals")

        return jsonify({'prediction': predicted_label})

    return jsonify({'error': 'Invalid file format'})

# Run the app
if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
