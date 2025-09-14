# src/nlp/data_processing_nlp.py
import json
import os
import pandas as pd

def load_cord19_contexts(json_dir, num_files=100):
    """
    Parses CORD-19 JSON files to extract text paragraphs (contexts).
    Returns a list of context strings.
    """
    contexts = []
    file_names = os.listdir(json_dir)[:num_files] # Limit for faster processing

    for file_name in file_names:
        if file_name.endswith('.json'):
            file_path = os.path.join(json_dir, file_name)
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Extract text from abstract and body
                for section in data.get('abstract', []):
                    contexts.append(section['text'])
                
                for section in data.get('body_text', []):
                    contexts.append(section['text'])
    
    print(f"Loaded {len(contexts)} contexts from {num_files} files.")
    return contexts

if __name__ == '__main__':
    # You will need to download the CORD-19 dataset's document_parses
    DATA_DIR = '../../data/cord19/document_parses/pdf_json'
    contexts = load_cord19_contexts(DATA_DIR)
    # In a real project, you would now format this into a question-answer dataset.
    # For simplicity, we are just extracting contexts.