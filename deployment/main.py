# deployment/main.py
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = FastAPI(title="xmedic.ai Diagnosis API")

# Load the trained model
MODEL_PATH = '../models/dermatology_model.h5'
model = load_model(MODEL_PATH)
IMAGE_SIZE = (100, 75)

# Define class names
class_names = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis-like lesions', 
               'Dermatofibroma', 'Melanocytic nevi', 'Melanoma', 'Vascular lesions']

def preprocess_image(image_bytes):
    """Preprocesses a single image for the model."""
    img = Image.open(io.BytesIO(image_bytes)).resize(IMAGE_SIZE)
    img_array = np.asarray(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    return img_array / 255.0

@app.post("/predict/dermatology")
async def predict_dermatology(file: UploadFile = File(...)):
    """Accepts an image file and returns a diagnosis prediction."""
    image_bytes = await file.read()
    processed_image = preprocess_image(image_bytes)
    
    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_name = class_names[predicted_class_index]
    confidence = float(np.max(prediction))

    return {
        "diagnosis": predicted_class_name,
        "confidence": f"{confidence*100:.2f}%"
    }

from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

# NEW: Import for NLP model
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
from pydantic import BaseModel

app = FastAPI(title="xmedic.ai Diagnosis and Literature API")

# --- Dermatology Model Loading (Existing) ---
DERMA_MODEL_PATH = '../models/dermatology_model.h5'
derma_model = load_model(DERMA_MODEL_PATH)
# ... (rest of the dermatology code) ...

# --- NLP Model Loading (New) ---
NLP_MODEL_PATH = '../models/biobert_qa_finetuned'
nlp_model = AutoModelForQuestionAnswering.from_pretrained(NLP_MODEL_PATH)
nlp_tokenizer = AutoTokenizer.from_pretrained(NLP_MODEL_PATH)

# Use a pipeline for easy inference
qa_pipeline = pipeline("question-answering", model=nlp_model, tokenizer=nlp_tokenizer)

class Query(BaseModel):
    question: str
    context: str

# --- Dermatology Endpoint (Existing) ---
@app.post("/predict/dermatology")
async def predict_dermatology(file: UploadFile = File(...)):
    # ... (existing prediction logic) ...
    pass

# --- Literature Query Endpoint (New) ---
@app.post("/query/literature")
async def query_literature(query: Query):
    """
    Accepts a clinical question and a context text, returns an answer.
    """
    result = qa_pipeline(question=query.question, context=query.context)
    return {
        "question": query.question,
        "answer": result['answer'],
        "confidence_score": result['score']
    }
