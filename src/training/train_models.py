
""" 
    This Python script defines functions to train text and visual recognition models, create a hybrid
    model, and ensure the existence of a directory for storing trained models.
"""
from pathlib import Path
from models.deep_learning.text_recognizer import TextRecognizer
from models.deep_learning.visual_recognizer import VisualRecognizer
from models.deep_learning.hybrid_recognizer import HybridRecognizer

def ensure_dir(path):
    """Ensure directory exists"""
    Path(path).mkdir(parents=True, exist_ok=True)

def train_text_model():
    print("Training text model...")
    text_model = TextRecognizer()
    text_model.save_model()
    print("Text model trained and saved")

def train_visual_model():
    print("Training visual model...")
    visual_model = VisualRecognizer()
    visual_model.save_model()
    print("Visual model trained and saved")

def create_hybrid_model():
    print("Creating hybrid model...")
    hybrid_model = HybridRecognizer()
    hybrid_model.save_models()
    print("Hybrid model created and components saved")

if __name__ == "__main__":
    # Ensure models directory exists
    ensure_dir("models/trained_models/")