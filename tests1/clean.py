import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("501 Bottle Dataset - Sheet1.csv")
# import os
# import cv2
# import numpy as np
# import pandas as pd
# import jellyfish
# from sklearn.neighbors import NearestNeighbors
# from sklearn.preprocessing import StandardScaler
# import easyocr
# from tensorflow.keras.applications import ResNet50, EfficientNetB0
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.resnet50 import preprocess_input
# from tensorflow.keras.models import Model
# import joblib
# from difflib import SequenceMatcher


# class WhiskyRecognizer:
#     def __init__(self, dataset_path="clean_image_whisky_dataset.csv", images_dir="images/"):
#         self.dataset = pd.read_csv(dataset_path)
#         self.images_dir = images_dir
#         self.reader = easyocr.Reader(['en'])
#         self.feature_extractor = self._init_feature_extractor()
#         self.knn_model = None
#         self.scaler = StandardScaler()
#         self._prepare_dataset()
#         self._build_feature_database()

#     def _init_feature_extractor(self):
#         """Initialize feature extractor with EfficientNet"""
#         base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
#         return Model(inputs=base_model.input, outputs=base_model.output)

#     def _preprocess_image(self, img):
#         """Enhanced image preprocessing"""
#         # Convert to grayscale and enhance contrast
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#         enhanced = clahe.apply(gray)
        
#         # Denoise and sharpen
#         denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
#         kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
#         sharpened = cv2.filter2D(denoised, -1, kernel)
        
#         # Convert back to 3 channels for CNN
#         return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

#     def _extract_shape_features(self, img):
#         """Extract bottle contour features"""
#         edges = cv2.Canny(img, 100, 200)
#         contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if contours:
#             largest = max(contours, key=cv2.contourArea)
#             x,y,w,h = cv2.boundingRect(largest)
#             return np.array([w/h, w*h])  # Aspect ratio and area
#         return np.zeros(2)

#     def _extract_visual_features(self, img_path):
#         """Enhanced feature extraction with shape features"""
#         img = cv2.imread(img_path)
#         if img is None:
#             return None
            
#         processed_img = self._preprocess_image(img)
        
#         # CNN Features (EfficientNet)
#         img_array = image.img_to_array(processed_img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array = preprocess_input(img_array)
#         cnn_features = self.feature_extractor.predict(img_array, verbose=0).flatten()
        
#         # SIFT Features
#         gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
#         sift = cv2.SIFT_create()
#         _, descriptors = sift.detectAndCompute(gray, None)
#         sift_features = np.mean(descriptors, axis=0) if descriptors is not None else np.zeros(128)
        
#         # Shape Features
#         shape_features = self._extract_shape_features(img)
        
#         # Weighted combination (adjust weights as needed)
#         return np.concatenate([
#             cnn_features * 0.4,      # CNN weight
#             sift_features * 0.4,     # SIFT weight
#             shape_features * 0.2     # Shape weight
#         ])

#     def _extract_text_features(self, img_path):
#         """Improved OCR with better preprocessing"""
#         img = cv2.imread(img_path)
#         if img is None:
#             return ""
            
#         # Enhanced preprocessing for OCR
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         blurred = cv2.GaussianBlur(gray, (5,5), 0)
#         _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
#         # Focus on center region (where label usually is)
#         h, w = thresholded.shape
#         roi = thresholded[h//4:3*h//4, w//4:3*w//4]
        
#         results = self.reader.readtext(roi)
#         return " ".join([res[1] for res in results])

#     # def _text_match_score(self, query_text, db_text):
#     #     """Better text comparison with exact match bonus"""
#     #     if not query_text or not db_text:
#     #         return 0
            
#     #     query_text = query_text.lower()
#     #     db_text = db_text.lower()
        
#     #     # Bonus for exact matches
#     #     if query_text in db_text:
#     #         return 1.0
#     #     return jellyfish.jaro_winkler(query_text, db_text)
    

#     def _text_match_score(self, query_text, db_text):
#         if not query_text or not db_text:
#             return 0
#         return SequenceMatcher(None, query_text.lower(), db_text.lower()).ratio()

#     def _build_feature_database(self):
#         """Build enhanced feature database"""
#         features = []
#         text_data = []
#         valid_indices = []
        
#         print("Building enhanced feature database...")
#         for idx, row in self.dataset.iterrows():
#             img_path = row['image_path']
            
#             visual_features = self._extract_visual_features(img_path)
#             if visual_features is None:
#                 continue
                
#             text = self._extract_text_features(img_path)
            
#             features.append(visual_features)
#             text_data.append(text)
#             valid_indices.append(idx)
            
#             if len(features) % 50 == 0:
#                 print(f"Processed {len(features)} images")
        
#         self.dataset = self.dataset.iloc[valid_indices]
#         self.dataset['ocr_text'] = text_data
        
#         features = np.array(features)
#         self.scaler.fit(features)
#         features_scaled = self.scaler.transform(features)
        
#         # Use more neighbors for better matching
#         self.knn_model = NearestNeighbors(n_neighbors=20, metric='cosine').fit(features_scaled)
#         self.features = features_scaled
#         print(f"Database built with {len(features)} entries")

#     def recognize_whisky(self, query_img_path, top_n=5):
#         """Hybrid recognition with visual and text matching"""
#         # Visual matching
#         query_features = self._extract_visual_features(query_img_path)
#         if query_features is None:
#             return None
            
#         query_features = self.scaler.transform([query_features])
#         distances, indices = self.knn_model.kneighbors(query_features)
        
#         # Text matching
#         query_text = self._extract_text_features(query_img_path)
        
#         # Combine results
#         results = []
#         for dist, idx in zip(distances[0], indices[0]):
#             item = self.dataset.iloc[idx]
#             text_score = self._text_match_score(query_text, item['ocr_text'])
            
#             # Combined score (adjust weights as needed)
#             combined_score = 0.7 * (1 - dist) + 0.3 * text_score
            
#             results.append({
#                 **item.to_dict(),
#                 'confidence': combined_score,
#                 'visual_sim': float(1 - dist),
#                 'text_sim': text_score
#             })
        
#         return sorted(results, key=lambda x: -x['confidence'])[:top_n]

#     def save_model(self, path="whisky_recognition_enhanced.pkl"):
#         joblib.dump({
#             'knn_model': self.knn_model,
#             'scaler': self.scaler,
#             'dataset': self.dataset,
#             'features': self.features
#         }, path)

#     @classmethod
#     def load_model(cls, path="whisky_recognition_enhanced.pkl"):
#         data = joblib.load(path)
#         recognizer = cls.__new__(cls)
#         recognizer.knn_model = data['knn_model']
#         recognizer.scaler = data['scaler']
#         recognizer.dataset = data['dataset']
#         recognizer.features = data['features']
#         recognizer.reader = easyocr.Reader(['en'])
#         recognizer.feature_extractor = recognizer._init_feature_extractor()
#         return recognizer

# # Example usage
# if __name__ == "__main__":
#     # Initialize or load model
#     try:
#         recognizer = WhiskyRecognizer.load_model()
#         print("Loaded pre-trained model")
#     except:
#         print("Building new model...")
#         recognizer = WhiskyRecognizer()
#         recognizer.save_model()
    
#     # Test recognition
#     test_img = "test_whisky2.jpg"
#     if os.path.exists(test_img):
#         results = recognizer.recognize_whisky(test_img)
        
#         print("\nTop Matches:")
#         for i, res in enumerate(results, 1):
#             print(f"{i}. {res['name']} (Confidence: {res['confidence']:.2f})")
#             print(f"   Visual: {res['visual_sim']:.2f}, Text: {res['text_sim']:.2f}")
#             print(f"   Type: {res['spirit_type']}, ABV: {res['abv']}%\n")
#     else:
#         print(f"Test image {test_img} not found")


# # 1. Drop rows with missing critical data (name, image_url)
# df = df.dropna(subset=["name", "image_url"])

# # 2. Clean 'name' column (remove extra quotes)
# df["name"] = df["name"].str.replace('"', '')

# # 3. Fill missing numerical values
# numerical_cols = ["proof", "abv", "avg_msrp", "fair_price", "shelf_price"]
# for col in numerical_cols:
#     if df[col].dtype in [np.float64, np.int64]:
#         df[col] = df[col].fillna(df[col].median())

# # 4. Standardize 'spirit_type'
# df["spirit_type"] = df["spirit_type"].str.strip().str.title()

# # 5. Remove duplicates (keep first occurrence)
# df = df.drop_duplicates(subset=["id", "name"], keep="first")

# # 6. Handle outliers (e.g., proof > 200)
# df = df[(df["proof"] > 0) & (df["proof"] <= 200)]

# # 7. (Optional) Extract age from name
# df["age"] = df["name"].str.extract(r'(\d+)\s*Year')  # Extracts "12" from "12 Year"

# # 8. Save cleaned data
# df.to_csv("cleaned_whisky_dataset.csv", index=False)


# import os
# import cv2
# import numpy as np
# import pandas as pd
# from sklearn.neighbors import NearestNeighbors
# from sklearn.preprocessing import StandardScaler
# import easyocr
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.resnet50 import preprocess_input
# from tensorflow.keras.models import Model
# import joblib

# class WhiskyRecognizer:
#     def __init__(self, dataset_path="dataset/processed/clean_image_whisky_dataset.csv", images_dir="images/"):
#         self.dataset = pd.read_csv(dataset_path)
#         self.images_dir = images_dir
#         self.reader = easyocr.Reader(['en'])  # Initialize OCR
        
#         # Initialize feature extractor first
#         self.feature_extractor = self._init_feature_extractor()
#         self.knn_model = None
#         self.scaler = StandardScaler()
        
#         # Prepare data
#         self._prepare_dataset()
#         self._build_feature_database()
        
#     def _init_feature_extractor(self):
#         """Initialize CNN feature extractor"""
#         base_model = ResNet50(weights='imagenet', include_top=False)
#         return Model(inputs=base_model.input, outputs=base_model.output)
    
#     def _prepare_dataset(self):
#         """Clean and prepare the dataset"""
#         self.dataset['image_path'] = self.dataset['id'].apply(
#             lambda x: f"{self.images_dir}{x}.jpg"
#         )
#         # Filter only existing images
#         self.dataset = self.dataset[self.dataset['image_path'].apply(os.path.exists)]
    
#     def _extract_visual_features(self, img_path):
#         """Extract visual features using CNN and traditional CV"""
#         try:
#             # Load and preprocess image
#             img = cv2.imread(img_path)
#             if img is None:
#                 print(f"Warning: Could not load image at {img_path}")
#                 return None
                
#             # Resize for CNN
#             img_resized = cv2.resize(img, (224, 224))
            
#             # CNN Features
#             img_array = image.img_to_array(img_resized)
#             img_array = np.expand_dims(img_array, axis=0)
#             img_array = preprocess_input(img_array)
#             cnn_features = self.feature_extractor.predict(img_array, verbose=0).flatten()
            
#             # Traditional CV Features
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             sift = cv2.SIFT_create()
#             _, descriptors = sift.detectAndCompute(gray, None)
#             sift_features = np.mean(descriptors, axis=0) if descriptors is not None else np.zeros(128)
            
#             # Combine features
#             return np.concatenate([cnn_features, sift_features])
#         except Exception as e:
#             print(f"Error processing image {img_path}: {str(e)}")
#             return None
    
#     def _extract_text_features(self, img_path):
#         """Extract text from label using OCR"""
#         try:
#             img = cv2.imread(img_path)
#             if img is None:
#                 return ""
            
#             # Preprocess for better OCR
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            
#             # Run OCR
#             results = self.reader.readtext(gray)
#             return " ".join([res[1] for res in results])
#         except Exception as e:
#             print(f"OCR failed for {img_path}: {str(e)}")
#             return ""
    
#     def _build_feature_database(self):
#         """Build feature database for all whiskies"""
#         features = []
#         text_data = []
#         valid_indices = []
        
#         print("Building feature database...")
#         for idx, row in self.dataset.iterrows():
#             img_path = row['image_path']
            
#             # Extract visual features
#             visual_features = self._extract_visual_features(img_path)
#             if visual_features is None:
#                 continue
                
#             # Extract text features
#             text = self._extract_text_features(img_path)
            
#             features.append(visual_features)
#             text_data.append(text)
#             valid_indices.append(idx)
            
#             if len(features) % 50 == 0:
#                 print(f"Processed {len(features)} images")
        
#         # Update dataset with only valid entries
#         self.dataset = self.dataset.iloc[valid_indices]
#         self.dataset['ocr_text'] = text_data
        
#         # Train KNN model
#         features = np.array(features)
#         self.scaler.fit(features)
#         features_scaled = self.scaler.transform(features)
#         self.knn_model = NearestNeighbors(n_neighbors=5, metric='cosine').fit(features_scaled)
        
#         # Save features for later use
#         self.features = features_scaled
#         print(f"Feature database built with {len(features)} entries")
    
#     def recognize_whisky(self, query_img_path, top_n=5):
#         """
#         Recognize a whisky from an image
#         Returns: List of matches with confidence scores
#         """
#         if not hasattr(self, 'knn_model'):
#             raise ValueError("Model not trained. Call _build_feature_database() first.")
            
#         # Extract query features
#         query_features = self._extract_visual_features(query_img_path)
#         if query_features is None:
#             return None
            
#         query_features = self.scaler.transform([query_features])
        
#         # Find nearest neighbors
#         distances, indices = self.knn_model.kneighbors(query_features, n_neighbors=top_n)
        
#         # Prepare results
#         results = []
#         for dist, idx in zip(distances[0], indices[0]):
#             match = self.dataset.iloc[idx].to_dict()
#             match['confidence'] = float(1 - dist)  # Convert distance to confidence
#             results.append(match)
        
#         return sorted(results, key=lambda x: -x['confidence'])
    
#     def save_model(self, path="whisky_recognition_model.pkl"):
#         """Save the trained model"""
#         if not hasattr(self, 'knn_model'):
#             raise ValueError("Model not trained. Nothing to save.")
            
#         joblib.dump({
#             'knn_model': self.knn_model,
#             'scaler': self.scaler,
#             'dataset': self.dataset,
#             'features': self.features
#         }, path)
#         print(f"Model saved to {path}")
    
#     @classmethod
#     def load_model(cls, path="models/feature_matching/whisky_recognition_model"):
#         """Load a trained model"""
#         data = joblib.load(path)
#         recognizer = cls.__new__(cls)
#         recognizer.knn_model = data['knn_model']
#         recognizer.scaler = data['scaler']
#         recognizer.dataset = data['dataset']
#         recognizer.features = data['features']
        
#         # Initialize other required attributes
#         recognizer.reader = easyocr.Reader(['en'])
#         recognizer.feature_extractor = recognizer._init_feature_extractor()
#         recognizer.images_dir = "images/"  # Default, can be updated if needed
        
#         return recognizer

# # Example Usage
# if __name__ == "__main__":
#     # Initialize or load the recognizer
#     try:
#         recognizer = WhiskyRecognizer.load_model()
#         print("Loaded pre-trained model")
#     except Exception as e:
#         print(f"Could not load model: {str(e)}")
#         print("Training new model...")
#         recognizer = WhiskyRecognizer()
#         recognizer.save_model()
    
#     # Test recognition
#     test_image_path = "test_whisky2.jpg"  # Replace with your test image
#     if os.path.exists(test_image_path):
#         results = recognizer.recognize_whisky(test_image_path)
        
#         if results:
#             print("\nTop Matches:")
#             for i, result in enumerate(results, 1):
#                 print(f"{i}. {result['name']} (Confidence: {result['confidence']:.2f})")
#                 print(f"   Type: {result['spirit_type']}, ABV: {result['abv']}%")
#                 print(f"   Brand ID: {result['brand_id']}, Image: {result['image_path']}\n")
#         else:
#             print("No matches found or error processing image")
#     else:
#         print(f"Test image not found at {test_image_path}")