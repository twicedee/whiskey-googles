import os
import cv2
import numpy as np
import pandas as pd
import easyocr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import joblib
import re

class TextRecognizer:
    def __init__(self, dataset_path="dataset/processed/clean_image_whisky_dataset.csv"):
        self.dataset = pd.read_csv(dataset_path)
        self.reader = easyocr.Reader(['en'])
        self.vectorizer = TfidfVectorizer()
        self._prepare_dataset()
        self._build_text_database()
        # self.whisky_abbreviations = {  
        #     "yo": "year old", "y.o": "year old", "yr": "year",
        #     "btch": "batch", "burbon": "bourbon"
        # }

    def _prepare_dataset(self):
        """Load and prepare text data"""
        self.dataset['text'] = self.dataset.apply(
            lambda row: self._extract_text(f"images/{row['id']}.jpg"), 
            axis=1
        )
        self.dataset = self.dataset.dropna(subset=['text'])
        
    def _extract_text(self, img_path):
        """Enhanced OCR with whisky-specific tuning"""
        img = cv2.imread(img_path)
        if img is None:
            return None
            
        # Dynamic label region (wider capture)
        h, w = img.shape[:2]
        roi = img[int(h*0.2):int(h*0.8), int(w*0.1):int(w*0.9)]
        
        # Advanced preprocessing
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=30)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16,16))
        enhanced = clahe.apply(denoised)
        _, thresholded = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR with whisky-optimized config
        results = self.reader.readtext(
            thresholded,
            batch_size=4,
            paragraph=True,
            min_size=20,
            contrast_ths=0.5,
            adjust_contrast=0.7,
            decoder='beamsearch'
        )
        
        # Safe text extraction with validation
        extracted_texts = []
        for res in results:
            try:
                # Check if result has at least 2 elements (text is at index 1)
                if len(res) >= 2:
                    # If confidence score is available (index 2) and > 0.5
                    if len(res) >= 3 and res[2] > 0.5:
                        extracted_texts.append(res[1])
                    # If no confidence score, just take the text
                    elif len(res) == 2:
                        extracted_texts.append(res[1])
            except (IndexError, TypeError):
                continue
                
        return " ".join(extracted_texts) if extracted_texts else None

    def _standardize_whisky_text(self, text):
        """Normalize whisky abbreviations"""
        text = text.lower()
        # for abbr, full in self.whisky_abbreviations.items():
        #     text = text.replace(abbr, full)
        return text

    def search(self, query_img_path, top_n=5):
        query_text = self._extract_text(query_img_path)
        if not query_text:
            return []
            
        # Compare against standardized dataset text
        query_vec = self.vectorizer.transform([query_text])
        sim_scores = cosine_similarity(query_vec, self.text_vectors)[0]
        
        top_indices = np.argsort(sim_scores)[-top_n:][::-1]
        return [{
            **self.dataset.iloc[idx].to_dict(),
            'confidence': float(sim_scores[idx]),
            'match_type': 'text'
        } for idx in top_indices]
    def _build_text_database(self):
        """Build TF-IDF vector database"""
        self.text_vectors = self.vectorizer.fit_transform(self.dataset['text'])


    def save_model(self, path="models/trained_models/text_model.pkl"):
        joblib.dump({
            'vectorizer': self.vectorizer,
            'dataset': self.dataset,
            'text_vectors': self.text_vectors 
        }, path)

    @classmethod
    def load_model(cls, path="models/trained_models/text_model.pkl"):
        data = joblib.load(path)
        model = cls.__new__(cls)
        model.vectorizer = data['vectorizer']
        model.dataset = data['dataset']
        model.text_vectors = data['text_vectors'] 
        model.reader = easyocr.Reader(['en'])
        return model
    
    
    
