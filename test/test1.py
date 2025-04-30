from models.deep_learning.hybrid_recognizer import HybridRecognizer


    
    
if __name__ == "__main__":
    # Initialize or load model
    recognizer = HybridRecognizer.load_models()

    results = recognizer.recognize("test_image.jpg")

    for i, res in enumerate(results, 1):
        print(f"{i}. {res['name']} (Confidence: {res['combined_score']:.2f})")
        print(f"   Type: {res['spirit_type']}, ABV: {res['abv']}%")
        print(f"   Match Type: {res['match_type']}\n")