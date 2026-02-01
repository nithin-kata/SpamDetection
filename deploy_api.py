"""
REST API deployment using FastAPI for the text classification system.
Alternative to Streamlit for API-based deployments.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
from typing import Dict, List
import uvicorn

# Import our modules
from src.preprocessing import TextPreprocessor
from src.models import TextClassifier

app = FastAPI(
    title="Text Classification API",
    description="API for text classification using machine learning",
    version="1.0.0"
)

# Global variables for loaded models
preprocessor = None
feature_extractor = None
classifier = None
label_names = None

class TextInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]
    processed_text: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

@app.on_event("startup")
async def load_model():
    """Load the trained model on startup."""
    global preprocessor, feature_extractor, classifier, label_names
    
    try:
        # Initialize preprocessor
        preprocessor = TextPreprocessor()
        
        # Try to load saved models
        model_files = [f for f in os.listdir("models") if f.endswith("_model.pkl")]
        
        if model_files:
            # Load the first available model
            model_file = model_files[0]
            model_name = model_file.replace("_model.pkl", "")
            
            # Load components
            classifier = TextClassifier("logistic_regression")  # Default
            classifier.load_model(f"models/{model_file}")
            
            feature_extractor = joblib.load(f"models/{model_name}_features.pkl")
            label_names = joblib.load(f"models/{model_name}_labels.pkl")
            
            print(f"Loaded model: {model_name}")
        else:
            print("No saved models found. Training a default model...")
            # Train a simple model for demo
            from data.dataset_loader import load_dataset
            
            texts, labels, label_names = load_dataset('movie_reviews')
            processed_texts = [preprocessor.preprocess_text(text) for text in texts]
            
            from src.preprocessing import FeatureExtractor
            feature_extractor = FeatureExtractor()
            features = feature_extractor.extract_tfidf_features(processed_texts, max_features=100)
            
            classifier = TextClassifier('logistic_regression')
            classifier.train(features, labels)
            
            print("Trained default model")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        # Set defaults
        preprocessor = TextPreprocessor()
        classifier = None
        feature_extractor = None
        label_names = {0: 'negative', 1: 'positive'}

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=classifier is not None
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_text(input_data: TextInput):
    """Predict the class of input text."""
    if not classifier or not feature_extractor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preprocess text
        processed_text = preprocessor.preprocess_text(input_data.text)
        
        # Extract features
        if hasattr(feature_extractor, 'tfidf_vectorizer') and feature_extractor.tfidf_vectorizer:
            features = feature_extractor.transform_tfidf([processed_text])
        else:
            features = feature_extractor.transform_bow([processed_text])
        
        # Make prediction
        prediction = classifier.predict(features)[0]
        probabilities = classifier.predict_proba(features)[0]
        
        # Format response
        predicted_class = label_names[prediction]
        confidence = float(probabilities[prediction])
        all_probs = {label_names[i]: float(prob) for i, prob in enumerate(probabilities)}
        
        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            all_probabilities=all_probs,
            processed_text=processed_text
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/models")
async def get_model_info():
    """Get information about the loaded model."""
    if not classifier:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": classifier.model_type,
        "classes": list(label_names.values()),
        "feature_extractor": "TF-IDF" if hasattr(feature_extractor, 'tfidf_vectorizer') else "BoW"
    }

@app.post("/batch_predict")
async def batch_predict(texts: List[str]):
    """Predict classes for multiple texts."""
    if not classifier or not feature_extractor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        for text in texts:
            # Preprocess
            processed_text = preprocessor.preprocess_text(text)
            
            # Extract features
            if hasattr(feature_extractor, 'tfidf_vectorizer') and feature_extractor.tfidf_vectorizer:
                features = feature_extractor.transform_tfidf([processed_text])
            else:
                features = feature_extractor.transform_bow([processed_text])
            
            # Predict
            prediction = classifier.predict(features)[0]
            probabilities = classifier.predict_proba(features)[0]
            
            results.append({
                "text": text,
                "predicted_class": label_names[prediction],
                "confidence": float(probabilities[prediction])
            })
        
        return {"predictions": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "deploy_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )