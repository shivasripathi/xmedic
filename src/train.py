# src/train.py
from data_processing import load_and_preprocess_data
from model import build_model

# --- Configuration ---
DATA_DIR = '../data/ham10000/images'
METADATA_PATH = '../data/ham10000/HAM10000_metadata.csv'
IMAGE_SIZE = (100, 75)
NUM_CLASSES = 7  # Number of lesion types
EPOCHS = 25
BATCH_SIZE = 32
MODEL_SAVE_PATH = '../models/dermatology_model.h5'

# 1. Load Data
X_train, X_test, y_train, y_test = load_and_preprocess_data(DATA_DIR, METADATA_PATH, IMAGE_SIZE)

# 2. Build Model
input_shape = (IMAGE_SIZE[1], IMAGE_SIZE[0], 3) # (height, width, channels)
model = build_model(input_shape, NUM_CLASSES)

# 3. Train Model
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test)
)

# 4. Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# 5. Save Model
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")