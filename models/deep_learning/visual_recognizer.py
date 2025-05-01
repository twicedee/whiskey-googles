import os
import cv2
import numpy as np
import pandas as pd
import logging
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from src.features.visual_features import detect_label_region
from sklearn.preprocessing import StandardScaler
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisualRecognizer:
    def __init__(self, dataset_path="dataset/processed/clean_image_whisky_dataset.csv"):
        """Initialize visual recognizer with data validation"""
        logger.info("Initializing VisualRecognizer")
        
        try:
            self.dataset = pd.read_csv(dataset_path)
            self.feature_extractor = self._init_feature_extractor()
            self.scaler = StandardScaler()
            self.label_encoder = LabelEncoder()
            self._prepare_dataset()
            self._build_visual_database()
            logger.info("VisualRecognizer initialized successfully")
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    def _init_feature_extractor(self):
        """Initialize EfficientNet feature extractor"""
        try:
            base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
            return Model(inputs=base_model.input, outputs=base_model.output)
        except Exception as e:
            logger.error(f"Failed to initialize feature extractor: {str(e)}")
            raise

    def _prepare_dataset(self):
        """Filter dataset for existing images with validation"""
        logger.info("Preparing dataset")
        
        if not hasattr(self, 'dataset') or self.dataset.empty:
            raise ValueError("Dataset is empty or not loaded properly")

        # Verify required columns exist
        required_columns = ['id', 'name', 'image_url']
        if not all(col in self.dataset.columns for col in required_columns):
            missing = [col for col in required_columns if col not in self.dataset.columns]
            raise ValueError(f"Dataset missing required columns: {missing}")

        # Verify images exist and are readable
        self.dataset = self.dataset[self.dataset['image_url'].apply(self._verify_image)]
    
        if self.dataset.empty:
            raise ValueError("No valid images found in the dataset")

    def _verify_image(self, img_path):
        """Verify image exists and is readable"""
        try:
            if not os.path.exists(img_path):
                logger.warning(f"Image not found: {img_path}")
                return False
                
            img = cv2.imread(img_path)
            if img is None or img.size == 0:
                logger.warning(f"Invalid image: {img_path}")
                return False
            return True
        except Exception as e:
            logger.warning(f"Error verifying image {img_path}: {str(e)}")
            return False
        
        
    def _extract_features(self, img_path):
        try:
            img = cv2.imread(img_path)
            if img is None:
                return None
                
            # Improved label detection with fallback
            label_img = detect_label_region(img)
            if label_img is not None:
                img = label_img
                # Add border to maintain aspect ratio
                img = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_CONSTANT)
            
            # Better preprocessing pipeline
            img = cv2.resize(img, (224, 224))  # Standard size for EfficientNet
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # EfficientNet expects RGB
            
            # Extract deep features
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            deep_features = self.feature_extractor.predict(img_array, verbose=0)[0]
            
            # Add shape and texture features
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            texture_feat = np.array([
                gray.mean(), gray.std(),
                sobelx.mean(), sobelx.std(),
                sobely.mean(), sobely.std()
            ])
            
            return np.concatenate([deep_features, texture_feat])
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            return None

    def _build_visual_database(self):
        features = []
        valid_indices = []
        
        for idx, row in self.dataset.iterrows():
            feat = self._extract_features(row['image_url'])
            if feat is not None and len(feat) > 0:
                features.append(feat)
                valid_indices.append(idx)
        
        self.dataset = self.dataset.iloc[valid_indices]
        self.features = self.scaler.fit_transform(np.array(features))
        
        # Replace k-NN with RandomForest
        self.clf = SVC(
                        C=1.0,
                        kernel='rbf',
                        probability=True,
                        class_weight='balanced'
                    )
        
        # Or use a neural network approach:
        
        
        model = Sequential([
            Dense(512, activation='relu', input_shape=(self.features.shape[1],)),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(len(self.dataset['name'].unique()), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        # Train on names rather than IDs to reduce class granularity
        
        y = self.label_encoder.fit_transform(self.dataset['name'])
        model.fit(self.features, y, epochs=20, batch_size=32)
        self.clf = model

    def search(self, query_img_path, top_n=5):
        query_feat = self._extract_features(query_img_path)
        if query_feat is None:
            return []
            
        query_feat = self.scaler.transform([query_feat])
        
        if isinstance(self.clf, Sequential):  # Neural network case
            probs = self.clf.predict(query_feat, verbose=0)[0]
            top_indices = np.argsort(probs)[-top_n:][::-1]
            names = self.label_encoder.inverse_transform(top_indices)
            return [{
                'name': name,
                'confidence': float(probs[idx]),
                'match_type': 'visual'
            } for idx, name in zip(top_indices, names)]
        else:  # Traditional classifier case
            probs = self.clf.predict_proba(query_feat)[0]
            top_indices = np.argsort(probs)[-top_n:][::-1]
            return [{
                **self.dataset.iloc[idx].to_dict(),
                'confidence': float(probs[idx]),
                'match_type': 'visual'
            } for idx in top_indices]

    def save_model(self, path="models/trained_models/visual_model.pkl"):
        """Save model with validation"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump({
                'clf': self.clf,
                'scaler': self.scaler,
                'dataset': self.dataset,
                'features': self.features
            }, path)
            logger.info(f"Model saved successfully to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise

    @classmethod
    def load_model(cls, path="models/trained_models/visual_model.pkl"):
        """Load model with validation"""
        try:
            data = joblib.load(path)
            model = cls.__new__(cls)
            model.clf = data['clf']
            model.scaler = data['scaler']
            model.dataset = data['dataset']
            model.features = data['features']
            model.feature_extractor = model._init_feature_extractor()
            
            # Initialize label encoder if it's a neural network model
            model.label_encoder = LabelEncoder()
            if isinstance(model.clf, Sequential) and 'name' in model.dataset.columns:
                model.label_encoder.fit(model.dataset['name'])
                
            logger.info(f"Model loaded successfully from {path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise