import os
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern, hog
import joblib

# Function to extract features from an image
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

# Load dataset and extract features
data = []
labels = []
classes = ['boots', 'flip_flops', 'sandals']
dataset_path = 'C:/Users/user/Documents/IU Data Science/From Model to Production/dataset/shoeTypeClassifierDataset/training'

for class_name in classes:
    class_path = os.path.join(dataset_path, class_name)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        features = extract_features(img_path)
        data.append(features)
        labels.append(class_name)

# Convert to numpy array
data = np.array(data)
labels = np.array(labels)

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler to the data (this computes the mean and standard deviation)
data = scaler.fit_transform(data)

# Save the scaler to a file
joblib.dump(scaler, 'scaler.pkl')

print("Scaler saved successfully.")
