# import pandas as pd
# import os
# from pathlib import Path
# import logging
# from tqdm import tqdm
# from sklearn.metrics import accuracy_score
# import json
# from models.deep_learning.text_recognizer import TextRecognizer
# from models.deep_learning.visual_recognizer import VisualRecognizer
# from models.deep_learning.hybrid_recognizer import HybridRecognizer


# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def evaluate_models_on_dataset(csv_path, images_dir, output_dir):
#     """
#     Evaluate text, visual, and hybrid models on the whisky dataset
    
#     Args:
#         csv_path (str): Path to the cleaned whisky dataset CSV
#         images_dir (str): Directory containing bottle images (named as id.jpg)
#         output_dir (str): Directory to save results
        
#     Returns:
#         dict: Evaluation metrics
#     """
#     try:
#         # Create output directory if it doesn't exist
#         os.makedirs(output_dir, exist_ok=True)
        
#         # Load dataset
#         logger.info("Loading dataset...")
#         df = pd.read_csv(csv_path)
        
#         # Initialize results dataframe
#         results = []
        
#         # Initialize accuracy counters
#         text_correct = 0
#         visual_correct = 0
#         hybrid_correct = 0
#         total = 0
        
#         # Initialize models (using your existing code)
#         logger.info("Initializing models...")
#         text_model = TextRecognizer.load_model('models/trained_models/text_model.pkl')
#         visual_model = VisualRecognizer.load_model('models/trained_models/visual_model.pkl')
#         hybrid_model = HybridRecognizer.load_models()
        
#         # Process each whisky in the dataset
#         logger.info("Processing images...")
#         for _, row in tqdm(df.iterrows(), total=len(df)):
#             try:
#                 image_path = Path(images_dir) / f"{row['id']}.jpg"
                
#                 if not os.path.exists(image_path):
#                     logger.warning(f"Image not found: {image_path}")
#                     continue
                
#                 # Get ground truth name
#                 true_name = row['name']
                
#                 # Get predictions from each model
#                 text_results = text_model.search(image_path)
#                 visual_results = visual_model.search(image_path)
#                 hybrid_results = hybrid_model.recognize(image_path)
                
#                 # Get top prediction from each model
#                 text_top = text_results[0].get('name', '') if text_results else ''
#                 visual_top = visual_results[0].get('name', '') if visual_results else ''
#                 hybrid_top = hybrid_results[0].get('name', '') if hybrid_results else ''
                
#                 # Get confidence/scores
#                 text_score = text_results[0].get('confidence', 0) if text_results else 0
#                 visual_score = visual_results[0].get('confidence', 0) if visual_results else 0
#                 hybrid_score = hybrid_results[0].get('combined_score', 0) if hybrid_results else 0
                
#                 # Check if predictions are correct
#                 text_correct += int(text_top == true_name)
#                 visual_correct += int(visual_top == true_name)
#                 hybrid_correct += int(hybrid_top == true_name)
#                 total += 1
                
#                 # Store results
#                 results.append({
#                     'id': row['id'],
#                     'name': true_name,
#                     'text_prediction': text_top,
#                     'text_score': text_score,
#                     'visual_prediction': visual_top,
#                     'visual_score': visual_score,
#                     'hybrid_prediction': hybrid_top,
#                     'hybrid_score': hybrid_score
#                 })
                
#             except Exception as e:
#                 logger.error(f"Error processing image {row['id']}: {str(e)}")
#                 continue
        
#         # Calculate accuracy metrics
#         metrics = {
#             'text_accuracy': text_correct / total if total > 0 else 0,
#             'visual_accuracy': visual_correct / total if total > 0 else 0,
#             'hybrid_accuracy': hybrid_correct / total if total > 0 else 0,
#             'total_samples': total
#         }
        
#         # Convert results to DataFrame
#         results_df = pd.DataFrame(results)
        
#         # Save results
#         results_path = os.path.join(output_dir, 'model_predictions.csv')
#         metrics_path = os.path.join(output_dir, 'evaluation_metrics.json')
        
#         results_df.to_csv(results_path, index=False)
#         with open(metrics_path, 'w') as f:
#             json.dump(metrics, f, indent=2)
        
#         logger.info(f"Results saved to {results_path}")
#         logger.info(f"Metrics saved to {metrics_path}")
        
#         # Print summary
#         print("\nEvaluation Summary:")
#         print(f"Text Model Accuracy: {metrics['text_accuracy']:.2%}")
#         print(f"Visual Model Accuracy: {metrics['visual_accuracy']:.2%}")
#         print(f"Hybrid Model Accuracy: {metrics['hybrid_accuracy']:.2%}")
#         print(f"Total Samples Evaluated: {metrics['total_samples']}")
        
#         return metrics
        
#     except Exception as e:
#         logger.error(f"Evaluation failed: {str(e)}")
#         raise

# if __name__ == "__main__":
#     # Configuration
#     CSV_PATH = "dataset/processed/clean_image_whisky_dataset.csv"
#     IMAGES_DIR = "images"  # Directory where images are stored as id.jpg
#     OUTPUT_DIR = "dataset/processed"
    
#     # Run evaluation
#     evaluate_models_on_dataset(CSV_PATH, IMAGES_DIR, OUTPUT_DIR)







import cv2
import logging
from pathlib import Path
import sys

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

# Import models
from models.deep_learning.text_recognizer import TextRecognizer
from models.deep_learning.visual_recognizer import VisualRecognizer
from models.deep_learning.hybrid_recognizer import HybridRecognizer

# Import feature extractors
from src.features.visual_features import extract_color_histogram, preprocess_image
from src.features.text_features import text_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def main():
    try:
        # Initialize models
        logger.info("Initializing models...")
        text_model = TextRecognizer.load_model('models/trained_models/text_model.pkl')
        visual_model = VisualRecognizer.load_model('models/trained_models/visual_model.pkl')
        hybrid_model = HybridRecognizer.load_models()
        
        # Example usage with sample image
        sample_image_path = "5979-large.webp"
        
        # Get predictions from all models
        logger.info("Running predictions...")
        text_results = text_model.search(sample_image_path)
        visual_results = visual_model.search(sample_image_path)
        hybrid_results = hybrid_model.recognize(sample_image_path)
        
        # Print results
        print("\nText Results (Top 2):")
        for res in text_results[:2]:
            print(f"- {res.get('name','Unknown')} (Confidence: {res.get('confidence',0):.2f})")
        
        print("\nVisual Results (Top 2):")
        for res in visual_results[:2]:
            print(f"- {res.get('name','Unknown')} (Confidence: {res.get('confidence',0):.2f}")
        
        print("\nHybrid Results (Top 2):")
        for res in hybrid_results[:2]:
            print(f"- {res.get('name','Unknown')} (Confidence: {res.get('combined_score',0):.2f})")
        
        # Example of using text similarity with results
        if text_results and len(text_results) >= 2:
            similarity = text_similarity(
                text_results[0].get('name', ''),
                text_results[1].get('name', '')
            )
            logger.info(f"Similarity between top 2 text results: {similarity:.2f}")
            
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()