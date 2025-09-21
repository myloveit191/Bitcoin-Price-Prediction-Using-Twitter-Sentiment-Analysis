#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing Package for Bitcoin Price Prediction
Tổng hợp các module preprocessing: text cleaning, segmentation, tokenization, embeddings

Author: Bitcoin Price Prediction Team
Date: 2025-09-20
"""

# Import main functions from each module
try:
    from .text_cleaning import clean_twitter_data, TwitterTextCleaner
    from .sentence_segmentation import segment_twitter_sentences, TwitterSentenceSegmenter
    from .tokenization import tokenize_twitter_data, TwitterTokenizer
    from .embeddings import generate_twitter_embeddings, TwitterEmbeddingsGenerator
    from .main import preprocess_twitter_data, TwitterPreprocessingPipeline
except ImportError as e:
    print(f"Warning: Could not import preprocessing modules: {e}")

# Package version
__version__ = "1.0.0"

# Package information
__author__ = "Bitcoin Price Prediction Team"
__email__ = "team@bitcoinprediction.com"
__description__ = "Advanced preprocessing pipeline for Twitter data in Bitcoin price prediction"

# Export main functions
__all__ = [
    # Main preprocessing function
    'preprocess_twitter_data',
    
    # Individual step functions
    'clean_twitter_data',
    'segment_twitter_sentences', 
    'tokenize_twitter_data',
    'generate_twitter_embeddings',
    
    # Classes
    'TwitterPreprocessingPipeline',
    'TwitterTextCleaner',
    'TwitterSentenceSegmenter',
    'TwitterTokenizer',
    'TwitterEmbeddingsGenerator'
]

# Package documentation
def get_preprocessing_info():
    """Get information about the preprocessing package"""
    info = {
        'package': 'preprocessing',
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'modules': [
            'text_cleaning - HTML decoding, emoji conversion, URL removal, noise reduction',
            'sentence_segmentation - NLTK tokenizer with Twitter-specific rules',
            'tokenization - Twitter-optimized tokenizer with crypto vocabulary',
            'embeddings - TF-IDF, Word2Vec, FastText, Sentence-BERT embeddings',
            'main - Complete preprocessing pipeline orchestrator'
        ],
        'main_functions': __all__
    }
    return info

def print_preprocessing_info():
    """Print preprocessing package information"""
    info = get_preprocessing_info()
    print(f"📦 {info['package'].upper()} PACKAGE")
    print("=" * 50)
    print(f"Version: {info['version']}")
    print(f"Author: {info['author']}")
    print(f"Description: {info['description']}")
    print("\n📋 Available Modules:")
    for module in info['modules']:
        print(f"  • {module}")
    print(f"\n🔧 Main Functions: {', '.join(info['main_functions'])}")
    print("=" * 50)

if __name__ == "__main__":
    print_preprocessing_info() 