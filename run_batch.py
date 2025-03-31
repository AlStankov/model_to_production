'''
This file searches for newly received image in the data folder, sends them in
batches to the API and stores the obtained classes
'''

import os
import requests
import json

# Configuration
IMAGE_FOLDER = os.path.join(os.getcwd(), 'new_products')  # Folder where new refund images are stored
API_URL = "http://127.0.0.1:5000"  # URL of the model service
RESULTS_FILE = os.path.join(os.getcwd(), 'results', 'classification_results.json')  # File to store classification results

# Ensure the 'results' folder exists
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

# Ensure the file exists (optional)
if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, 'w') as file:
        file.write('{}')  # Writing an empty JSON object

def get_image_files(folder):
    """Returns a list of image file paths in the given folder."""
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

def classify_images():
    """Sends images in batch to the classification API and stores the results."""
    image_files = get_image_files(IMAGE_FOLDER)
    results = []

    if not image_files:
        print("No new images found.")
        return

    for image_path in image_files:
        with open(image_path, "rb") as img_file:
            files = {"file": img_file}
            response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                prediction = response.json()
                results.append({"image": os.path.basename(image_path), "prediction": prediction})
            else:
                print(f"Error processing {image_path}: {response.status_code}")

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Classification completed. Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    classify_images()

