# src/data_processing.py
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import os

# Define lesion types for classification
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

def load_and_preprocess_data(data_dir, metadata_path, image_size=(100, 75)):
    """Loads images, preprocesses them, and splits into train/test sets."""
    df = pd.read_csv(metadata_path)
    df['lesion_type'] = df['dx'].map(lesion_type_dict.get)
    df['image_path'] = df['image_id'].apply(lambda x: os.path.join(data_dir, f'{x}.jpg'))

    # Load images
    images = []
    for path in df['image_path']:
        img = Image.open(path).resize(image_size)
        images.append(np.asarray(img))

    X = np.array(images)
    y = pd.get_dummies(df['dx']).values # One-hot encode labels

    # Normalize images
    X = X / 255.0

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Data loaded. Training shape: {X_train.shape}, Testing shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    # Example usage (assuming data is in the correct folder)
    DATA_DIR = '../data/ham10000/images'
    METADATA_PATH = '../data/ham10000/HAM10000_metadata.csv'
    load_and_preprocess_data(DATA_DIR, METADATA_PATH)