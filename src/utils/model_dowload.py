"""
Utility script to download a pre-trained model and tokenizer from the 
Hugging Face Hub and save them to a local directory for offline use.

This script should be run once to populate the local model directory.
After running, other scripts (like the training script) can point to the
local path instead of the Hugging Face model name.
"""
print("--- Checkpoint 1: Script execution started ---")

import os
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

print("--- Checkpoint 2: Imports successful, proceeding to run... ---")

# --- Configuration ---
# Specify the model you want to download. We'll use the QA model from your
# training script as the primary example.
MODEL_TO_DOWNLOAD = "dmis-lab/biobert-base-cased-v1.1-squad"

# --- MODIFIED SECTION ---
# Define the local directory where the model will be saved.
# We construct the path dynamically to be relative to the project root,
# which avoids pathing errors and makes the script portable.
# This finds the directory of the current script (utils)
script_dir = os.path.dirname(os.path.abspath(__file__))
# This goes up two levels to get to the project root (xmedic_ai/)
project_root = os.path.dirname(os.path.dirname(script_dir))
# This creates the final, correct path: 'xmedic_ai/models/local_biobert_qa'
LOCAL_SAVE_PATH = os.path.join(project_root, "models", "local_biobert_qa")


def download_model_locally():
    """
    Fetches the tokenizer and model from the Hugging Face Hub and saves
    them to the specified local directory.
    """
    print(f"--- Starting Download for Model: {MODEL_TO_DOWNLOAD} ---")
    print(f"Models will be saved to: {LOCAL_SAVE_PATH}")

    # Create the target directory if it doesn't already exist
    if not os.path.exists(LOCAL_SAVE_PATH):
        os.makedirs(LOCAL_SAVE_PATH)
        print(f"Created directory: {LOCAL_SAVE_PATH}")

    # 1. Download and save the tokenizer
    # The .from_pretrained() method downloads the files from the hub.
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_TO_DOWNLOAD)
    
    # The .save_pretrained() method saves the downloaded files to a local path.
    tokenizer.save_pretrained(LOCAL_SAVE_PATH)
    print(f"Tokenizer saved successfully to {LOCAL_SAVE_PATH}")

    # 2. Download and save the model
    print("\nDownloading model (this may take a few minutes)...")
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_TO_DOWNLOAD)
    model.save_pretrained(LOCAL_SAVE_PATH)
    print(f"Model saved successfully to {LOCAL_SAVE_PATH}")


def test_loading_from_local():
    """
    Verifies that the model can be loaded correctly from the local directory,
    confirming the download was successful.
    """
    print("\n--- Verifying Local Model ---")
    try:
        # Load the tokenizer and model by pointing to the local directory path
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_SAVE_PATH)
        model = AutoModelForQuestionAnswering.from_pretrained(LOCAL_SAVE_PATH)
        print("Successfully loaded tokenizer and model from the local directory!")
        print(f"You can now use the path '{LOCAL_SAVE_PATH}' in your training script.")
    except Exception as e:
        print(f"Error: Failed to load model from local directory. {e}")


if __name__ == "__main__":
    # Execute the download and verification process
    download_model_locally()
    test_loading_from_local()