import pandas as pd
import os
import sys
from pathlib import Path
import logging
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import json
from models.deep_learning.text_recognizer import TextRecognizer
from models.deep_learning.visual_recognizer import VisualRecognizer
from models.deep_learning.hybrid_recognizer import HybridRecognizer
from src.features.text_features import text_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_models_on_dataset(csv_path, images_dir, output_dir):
    """
    Evaluate text, visual, and hybrid models on the whisky dataset
    
    Args:
        csv_path (str): Path to the cleaned whisky dataset CSV
        images_dir (str): Directory containing bottle images (named as id.jpg)
        output_dir (str): Directory to save results
        
    Returns:
        dict: Evaluation metrics
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dataset
        logger.info("Loading dataset...")
        df = pd.read_csv(csv_path)
        
        # Initialize results dataframe
        results = []
        
        # Initialize accuracy counters
        text_correct = 0
        visual_correct = 0
        hybrid_correct = 0
        total = 0
        
        # Initialize models
        logger.info("Initializing models...")
        text_model = TextRecognizer.load_model('models/trained_models/text_model.pkl')
        visual_model = VisualRecognizer.load_model('models/trained_models/visual_model.pkl')
        hybrid_model = HybridRecognizer.load_models()
        
        # Process each whisky in the dataset
        logger.info("Processing images...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            try:
                image_path = Path(images_dir) / f"{row['id']}.jpg"
                
                if not image_path.exists():
                    logger.warning(f"Image not found: {image_path}")
                    continue
                
                # Get ground truth name
                true_name = row['name']
                
                # Get predictions from each model (with error handling)
                try:
                    text_results = text_model.search(image_path) or []
                    visual_results = visual_model.search(image_path) or []
                    hybrid_results = hybrid_model.recognize(image_path) or []
                except Exception as e:
                    logger.error(f"Prediction failed for image {row['id']}: {str(e)}")
                    continue
                
                # Get top prediction from each model
                text_top = text_results[0].get('name', '') if text_results else ''
                visual_top = visual_results[0].get('name', '') if visual_results else ''
                hybrid_top = hybrid_results[0].get('name', '') if hybrid_results else ''
                
                # Get confidence/scores
                text_score = text_results[0].get('confidence', 0) if text_results else 0
                visual_score = visual_results[0].get('confidence', 0) if visual_results else 0
                hybrid_score = hybrid_results[0].get('combined_score', 0) if hybrid_results else 0
                
                # Check if predictions are correct
                text_correct += int(text_top == true_name)
                visual_correct += int(visual_top == true_name)
                hybrid_correct += int(hybrid_top == true_name)
                total += 1
                
                # Store results
                results.append({
                    'id': row['id'],
                    'name': true_name,
                    'text_prediction': text_top,
                    'text_score': text_score,
                    'visual_prediction': visual_top,
                    'visual_score': visual_score,
                    'hybrid_prediction': hybrid_top,
                    'hybrid_score': hybrid_score,
                    'text_results_count': len(text_results),
                    'visual_results_count': len(visual_results),
                    'hybrid_results_count': len(hybrid_results)
                })
                
            except Exception as e:
                logger.error(f"Error processing image {row['id']}: {str(e)}")
                continue
        
        # Calculate accuracy metrics
        metrics = {
            'text_accuracy': text_correct / total if total > 0 else 0,
            'visual_accuracy': visual_correct / total if total > 0 else 0,
            'hybrid_accuracy': hybrid_correct / total if total > 0 else 0,
            'total_samples': total,
            'average_text_results': sum(r['text_results_count'] for r in results) / len(results) if results else 0,
            'average_visual_results': sum(r['visual_results_count'] for r in results) / len(results) if results else 0,
            'average_hybrid_results': sum(r['hybrid_results_count'] for r in results) / len(results) if results else 0
        }
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        results_path = Path(output_dir) / 'model_predictions.csv'
        metrics_path = Path(output_dir) / 'evaluation_metrics.json'
        
        results_df.to_csv(results_path, index=False)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
        logger.info(f"Metrics saved to {metrics_path}")
        
        # Print summary
        print("\nEvaluation Summary:")
        print(f"Text Model Accuracy: {metrics['text_accuracy']:.2%}")
        print(f"Visual Model Accuracy: {metrics['visual_accuracy']:.2%}")
        print(f"Hybrid Model Accuracy: {metrics['hybrid_accuracy']:.2%}")
        print(f"Total Samples Evaluated: {metrics['total_samples']}")
        print(f"\nAverage predictions per sample:")
        print(f"Text: {metrics['average_text_results']:.1f}")
        print(f"Visual: {metrics['average_visual_results']:.1f}")
        print(f"Hybrid: {metrics['average_hybrid_results']:.1f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise
    
def predict_single_image(image_path=None):
    """Predict single image with all models"""
    try:
        if image_path is None:
            image_path = input("Enter path to image file: ").strip()
        
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Initialize models
        logger.info("Initializing models...")
        text_model = TextRecognizer.load_model('models/trained_models/text_model.pkl')
        visual_model = VisualRecognizer.load_model('models/trained_models/visual_model.pkl')
        hybrid_model = HybridRecognizer.load_models()
        
        # Get predictions from all models
        logger.info("Running predictions...")
        text_results = text_model.search(image_path) or []
        visual_results = visual_model.search(image_path) or []
        hybrid_results = hybrid_model.recognize(image_path) or []
        
        # Print results
        def print_results(results, model_name, score_key='confidence', limit=3):
            print(f"\n{model_name} Results (Top {limit}):")
            if not results:
                print("No predictions returned")
                return
                
            for i, res in enumerate(results[:limit], 1):
                name = res.get('name', 'Unknown')
                score = res.get(score_key, 0)
                print(f"{i}. {name} (Score: {score:.2f})")
        
        print_results(text_results, "Text")
        print_results(visual_results, "Visual")
        print_results(hybrid_results, "Hybrid", 'combined_score')
        
        # Calculate text similarity if available
        if len(text_results) >= 2:
            similarity = text_similarity(
                text_results[0].get('name', ''),
                text_results[1].get('name', '')
            )
            print(f"\nText similarity between top 2 results: {similarity:.2f}")
            
        return {
            'text': text_results,
            'visual': visual_results,
            'hybrid': hybrid_results
        }
            
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return {
            'error': str(e),
            'text': [],
            'visual': [],
            'hybrid': []
        }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Whiskey Recognition System")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate models on full dataset')
    eval_parser.add_argument('--csv', required=True, help='Path to dataset CSV')
    eval_parser.add_argument('--images', required=True, help='Path to images directory')
    eval_parser.add_argument('--output', required=True, help='Output directory for results')
    
    # Prediction command
    pred_parser = subparsers.add_parser('predict', help='Predict single image')
    pred_parser.add_argument('--image', help='Path to image file (optional)')
    
    args = parser.parse_args()
    
    if args.command == 'evaluate':
        evaluate_models_on_dataset(args.csv, args.images, args.output)
    elif args.command == 'predict':
        predict_single_image(args.image)
    else:
        parser.print_help()