from .confidence import apply_confidence_threshold

def get_filtered_predictions(image_path, text_model, visual_model, hybrid_model):
    """
    The function `get_filtered_predictions` takes image predictions from text, visual, and hybrid
    models, applies confidence thresholds, and returns filtered predictions for each model.
    
    :return: The function `get_filtered_predictions` returns a dictionary containing filtered
    predictions for text, visual, and hybrid models based on the confidence thresholds set for each
    model.
    """
    
    # Define confidence thresholds 
    TEXT_THRESHOLD = 0.3
    VISUAL_THRESHOLD = 0.7
    HYBRID_THRESHOLD = 0.5
    
    # Get raw predictions
    text_results = text_model.search(image_path)
    visual_results = visual_model.search(image_path)
    hybrid_results = hybrid_model.recognize(image_path)
    
    # Apply confidence thresholds
    filtered_text = apply_confidence_threshold(text_results, TEXT_THRESHOLD)
    filtered_visual = apply_confidence_threshold(visual_results, VISUAL_THRESHOLD)
    filtered_hybrid = apply_confidence_threshold(hybrid_results, HYBRID_THRESHOLD)
    
    return {
        'text': filtered_text,
        'visual': filtered_visual,
        'hybrid': filtered_hybrid
    }