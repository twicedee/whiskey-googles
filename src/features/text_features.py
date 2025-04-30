import jellyfish
import re

def text_similarity(query, target):
    """Whisky-optimized text similarity"""
    if not query or not target:
        return 0
        
    # Standardize text
    query = query.lower().strip()
    target = target.lower().strip()
    
    # Handle common whisky patterns
    for pattern in [r'(\d+)\s*yo', r'(\d+)\s*yr']:
        query = re.sub(pattern, r'\1 year old', query)
        target = re.sub(pattern, r'\1 year old', target)
    
    # Combined metrics
    jw = jellyfish.jaro_winkler_similarity(query, target)
    lev = 1 - (jellyfish.levenshtein_distance(query, target) / max(len(query), len(target)))
    
    # Weighted score favoring name matches
    base_score = (jw * 0.6 + lev * 0.4)
    
    # Boost exact year matches
    query_years = set(re.findall(r'(?:19|20)\d{2}', query))
    target_years = set(re.findall(r'(?:19|20)\d{2}', target))
    if query_years and query_years == target_years:
        base_score *= 1.2
        
    return min(base_score, 1.0)  # Cap at 1.0