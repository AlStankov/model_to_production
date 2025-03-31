'''
This files contains the ML model for image classification
'''
import pickle
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler


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

# Load dataset
data = []
labels = []
classes = ['boots', 'flip_flops', 'sandals']
dataset_path = os.path.join(os.getcwd(), 'dataset', 'shoeTypeClassifierDataset', 'training')

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

# Normalize features
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)

# Train SVM classifier with improved hyperparameters
svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
svm.fit(X_train, y_train)

# Evaluate the model
y_pred = svm.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred, zero_division=0))

# Save the model as a pickle file
with open('image_classifier.pkl', 'wb') as file:
    pickle.dump(svm, file)


