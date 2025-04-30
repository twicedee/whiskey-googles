from models.deep_learning.text_recognizer import TextRecognizer
from models.deep_learning.visual_recognizer import VisualRecognizer
import numpy as np

class HybridRecognizer:
    def __init__(self):
        self.text_model = TextRecognizer()
        self.visual_model = VisualRecognizer()
    def recognize(self, query_img_path, top_n=5):
        text_results = self.text_model.search(query_img_path, top_n*3)
        visual_results = self.visual_model.search(query_img_path, top_n*3)
        
        # Calculate modality quality
        text_quality = np.mean([r['confidence'] for r in text_results[:3]]) if text_results else 0
        visual_quality = np.mean([r['confidence'] for r in visual_results[:3]]) if visual_results else 0
        
        # Dynamic weights
        text_weight = 0.7 if text_quality > 0.4 else 0.3
        visual_weight = 1 - text_weight
        
        # Combine with normalization
        max_text_conf = max([r['confidence'] for r in text_results], default=1)
        max_visual_conf = max([r['confidence'] for r in visual_results], default=1)
        
        combined = []
        for result in text_results + visual_results:
            if result['match_type'] == 'text':
                norm_conf = result['confidence'] / max_text_conf
                score = norm_conf * text_weight
            else:
                norm_conf = result['confidence'] / max_visual_conf
                score = norm_conf * visual_weight
            
            # Create a unique key for deduplication that doesn't rely on 'id'
            unique_key = (
                result.get('id', hash(result['name'])),  # Use id if exists, else hash of name
                result['name'],
                result['match_type']
            )
            
            combined.append({
                **result,
                'combined_score': score,
                '_unique_key': unique_key
            })
        
        # Deduplicate using our composite key and sort
        unique_results = {r['_unique_key']: r for r in combined}.values()
        return sorted(unique_results, key=lambda x: -x['combined_score'])[:top_n]

    def save_models(self):
        self.text_model.save_model()
        self.visual_model.save_model()

    @classmethod
    def load_models(cls):
        model = cls.__new__(cls)
        model.text_model = TextRecognizer.load_model()
        model.visual_model = VisualRecognizer.load_model()
        return model
    
    
