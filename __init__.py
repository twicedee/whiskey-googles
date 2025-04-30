"""
whisky_goggles package - Whisky recognition system
"""
from . import models
from . import src

# Version of the package
__version__ = "0.1.0"

from .models.deep_learning.text_recognizer import TextRecognizer
from .models.deep_learning.visual_recognizer import VisualRecognizer
from .models.deep_learning.hybrid_recognizer import HybridRecognizer

__all__ = ['TextRecognizer', 'VisualRecognizer', 'HybridRecognizer']