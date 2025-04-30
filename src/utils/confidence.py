def apply_confidence_threshold(results, min_confidence=0.5, fallback_name="Unknown"):
    """
    Filters prediction results based on confidence threshold.
    
    Args:
        results: List of prediction dicts (must contain 'confidence'/'combined_score')
        min_confidence: Minimum required confidence score (0-1)
        fallback_name: Name to return when no results meet threshold
        
    Returns:
        dict: Either the best result above threshold or a fallback dict
    """
    if not results:
        return {'name': fallback_name, 'confidence': 0.0, 'match_type': 'none'}
    
    # Use 'combined_score' for hybrid results or 'confidence' for single-model
    score_key = 'combined_score' if 'combined_score' in results[0] else 'confidence'
    
    # Get best result
    best_result = max(results, key=lambda x: x[score_key])
    
    if best_result[score_key] >= min_confidence:
        return best_result
    else:
        return {
            'name': fallback_name,
            'confidence': best_result[score_key],
            'match_type': 'below_threshold',
            'original_top_match': best_result['name']  # For debugging
        }