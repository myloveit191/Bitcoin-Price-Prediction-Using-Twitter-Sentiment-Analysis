#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Preprocessing Module for Bitcoin Price Prediction
Tổng hợp toàn bộ pipeline preprocessing: text cleaning, tokenization, embeddings

Author: Bitcoin Price Prediction Team
Date: 2025-09-20
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import local preprocessing modules
from text_cleaning import clean_twitter_data
from tokenization import tokenize_twitter_data
from embeddings import generate_twitter_embeddings

class TwitterPreprocessingPipeline:
    """Complete preprocessing pipeline for Twitter data"""
    
    def __init__(self, config=None):
        """
        Initialize preprocessing pipeline
        
        Args:
            config: Configuration dictionary with preprocessing parameters
        """
        self.config = config or {}
        self.results = {}
        self.statistics = {}
        
        # Default configuration
        self.default_config = {
            'text_cleaning': {
                'decode_html': True,
                'convert_emojis': True,
                'remove_urls': True,
                'handle_mentions': True,
                'handle_hashtags': True,
                'remove_rt': True,
                'remove_noise': True,
                'normalize': True,
                'keep_hashtags': True,
                'keep_mentions': False
            },
            'tokenization': {
                'preserve_crypto': True,
                'remove_stopwords': True,
                'remove_mentions': True,
                'remove_hashtags': False,
                'remove_urls': True,
                'remove_numbers': True,
                'remove_punctuation': True,
                'min_length': 2,
                'max_length': 50
            },
            'embeddings': {
                'embedding_types': ['tfidf', 'word2vec', 'fasttext'],
                'save_embeddings': True,
                'save_dir': 'embeddings',
                'tfidf_params': {
                    'max_features': 5000,
                    'min_df': 2,
                    'max_df': 0.95,
                    'ngram_range': (1, 2)
                },
                'word2vec_params': {
                    'vector_size': 100,
                    'window': 5,
                    'min_count': 2,
                    'epochs': 10
                },
                'fasttext_params': {
                    'vector_size': 100,
                    'window': 5,
                    'min_count': 2,
                    'epochs': 10
                }
            },
            'output': {
                'save_intermediate': True,
                'save_final': True,
                'output_dir': 'preprocessed_data'
            }
        }
        
        # Merge user config with defaults
        self._merge_config()
    
    def _merge_config(self):
        """Merge user configuration with defaults"""
        for key, default_value in self.default_config.items():
            if key not in self.config:
                self.config[key] = default_value
            elif isinstance(default_value, dict):
                # Merge nested dictionaries
                for sub_key, sub_value in default_value.items():
                    if sub_key not in self.config[key]:
                        self.config[key][sub_key] = sub_value
    
    def step1_text_cleaning(self, df, text_column='text'):
        """
        Step 1: Clean Twitter text data
        
        Args:
            df: Input DataFrame with raw Twitter data
            text_column: Name of the text column to clean
            
        Returns:
            Cleaned DataFrame
        """
        print("\n" + "="*60)
        print("🧹 STEP 1: TEXT CLEANING")
        print("="*60)
        
        if df.empty:
            print("   ❌ Empty DataFrame provided")
            return df
        
        print(f"   📊 Input: {len(df)} tweets")
        print(f"   📝 Text column: '{text_column}'")
        
        # Perform text cleaning
        cleaned_df, cleaning_stats = clean_twitter_data(
            df, text_column, **self.config['text_cleaning']
        )
        
        # Store results
        self.results['cleaned_data'] = cleaned_df
        self.statistics['text_cleaning'] = cleaning_stats
        
        # Save intermediate results if requested
        if self.config['output']['save_intermediate']:
            self._save_intermediate_data(cleaned_df, 'cleaned_data')
        
        print(f"   ✅ Output: {len(cleaned_df)} cleaned tweets")
        return cleaned_df
    
    def step2_tokenization(self, df, text_column='text_cleaned'):
        """
        Step 2: Tokenize cleaned text
        
        Args:
            df: DataFrame with cleaned text
            text_column: Name of the cleaned text column
            
        Returns:
            DataFrame with tokenized text
        """
        print("\n" + "="*60)
        print("🔤 STEP 2: TOKENIZATION")
        print("="*60)
        
        if df.empty:
            print("   ❌ Empty DataFrame provided")
            return df
        
        print(f"   📊 Input: {len(df)} cleaned texts")
        print(f"   📝 Text column: '{text_column}'")
        
        # Perform tokenization
        tokenized_df, tokenization_stats = tokenize_twitter_data(
            df, text_column, **self.config['tokenization']
        )
        
        # Store results
        self.results['tokenized_data'] = tokenized_df
        self.statistics['tokenization'] = tokenization_stats
        
        # Save intermediate results if requested
        if self.config['output']['save_intermediate']:
            self._save_intermediate_data(tokenized_df, 'tokenized_data')
        
        print(f"   ✅ Output: {len(tokenized_df)} tokenized texts")
        return tokenized_df
    
    def step3_embeddings_generation(self, df, tokens_column='tokens'):
        """
        Step 3: Generate embeddings from tokenized text
        
        Args:
            df: DataFrame with tokenized text
            tokens_column: Name of the tokens column
            
        Returns:
            Dictionary with embeddings and models
        """
        print("\n" + "="*60)
        print("🎯 STEP 3: EMBEDDINGS GENERATION")
        print("="*60)
        
        if df.empty:
            print("   ❌ Empty DataFrame provided")
            return {}
        
        print(f"   📊 Input: {len(df)} tokenized texts")
        print(f"   📝 Tokens column: '{tokens_column}'")
        
        # Generate embeddings
        embeddings_results = generate_twitter_embeddings(
            df, tokens_column, **self.config['embeddings']
        )
        
        # Store results
        self.results['embeddings'] = embeddings_results
        if 'statistics' in embeddings_results:
            self.statistics['embeddings'] = embeddings_results['statistics']
        
        print(f"   ✅ Generated embeddings for {len(df)} texts")
        return embeddings_results
    
    def run_complete_pipeline(self, df, text_column='text'):
        """
        Run the complete preprocessing pipeline
        
        Args:
            df: Input DataFrame with raw Twitter data
            text_column: Name of the text column
            
        Returns:
            Dictionary containing all results and statistics
        """
        print("🚀 TWITTER PREPROCESSING PIPELINE")
        print("=" * 80)
        print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Input data: {len(df)} tweets")
        print(f"🎯 Goal: Complete preprocessing of Twitter data")
        print("=" * 80)
        
        try:
            # Step 1: Text Cleaning
            cleaned_df = self.step1_text_cleaning(df, text_column)
            
            if cleaned_df.empty:
                print("\n❌ Pipeline stopped: No data after text cleaning")
                return self._prepare_final_results()
            
            # Step 2: Tokenization
            tokenized_df = self.step2_tokenization(cleaned_df)
            
            if tokenized_df.empty:
                print("\n❌ Pipeline stopped: No data after tokenization")
                return self._prepare_final_results()
            
            # Step 3: Embeddings Generation
            embeddings_results = self.step3_embeddings_generation(tokenized_df)
            
            # Prepare final results
            final_results = self._prepare_final_results()
            
            # Save final results if requested
            if self.config['output']['save_final']:
                self._save_final_results(final_results)
            
            print("\n" + "="*80)
            print("🎉 PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"📅 Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self._print_final_statistics()
            print("="*80)
            
            return final_results
            
        except Exception as e:
            print(f"\n❌ Pipeline failed with error: {e}")
            import traceback
            traceback.print_exc()
            return self._prepare_final_results()
    
    def _prepare_final_results(self):
        """Prepare final results dictionary"""
        return {
            'data': self.results,
            'statistics': self.statistics,
            'config': self.config,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _save_intermediate_data(self, df, data_name):
        """Save intermediate data to CSV"""
        output_dir = self.config['output']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{data_name}_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        
        try:
            df.to_csv(filepath, index=False, encoding='utf-8')
            print(f"   💾 Saved intermediate data: {filepath}")
        except Exception as e:
            print(f"   ⚠️ Failed to save intermediate data: {e}")
    
    def _save_final_results(self, results):
        """Save final results"""
        output_dir = self.config['output']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save final tokenized data
        if 'tokenized_data' in self.results:
            filename = f"final_preprocessed_data_{timestamp}.csv"
            filepath = os.path.join(output_dir, filename)
            try:
                self.results['tokenized_data'].to_csv(filepath, index=False, encoding='utf-8')
                print(f"   💾 Saved final preprocessed data: {filepath}")
            except Exception as e:
                print(f"   ⚠️ Failed to save final data: {e}")
        
        # Save statistics
        stats_filename = f"preprocessing_statistics_{timestamp}.txt"
        stats_filepath = os.path.join(output_dir, stats_filename)
        try:
            with open(stats_filepath, 'w', encoding='utf-8') as f:
                f.write("TWITTER PREPROCESSING STATISTICS\n")
                f.write("=" * 50 + "\n")
                f.write(f"Timestamp: {results['timestamp']}\n\n")
                
                for step, stats in results['statistics'].items():
                    f.write(f"{step.upper()}:\n")
                    for key, value in stats.items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")
            
            print(f"   💾 Saved preprocessing statistics: {stats_filepath}")
        except Exception as e:
            print(f"   ⚠️ Failed to save statistics: {e}")
    
    def _print_final_statistics(self):
        """Print final statistics summary"""
        print("📊 FINAL STATISTICS:")
        
        if 'text_cleaning' in self.statistics:
            stats = self.statistics['text_cleaning']
            print(f"   🧹 Text Cleaning: {stats.get('final_text_count', 'N/A')} texts")
        
        if 'tokenization' in self.statistics:
            stats = self.statistics['tokenization']
            print(f"   🔤 Tokenization: {stats.get('total_tokens', 'N/A')} tokens")
            print(f"      - Vocabulary size: {stats.get('vocabulary_size', 'N/A')}")
        
        if 'embeddings' in self.statistics:
            stats = self.statistics['embeddings']
            print("   🎯 Embeddings generated:")
            for emb_type, emb_stats in stats.items():
                if isinstance(emb_stats, dict) and 'shape' in emb_stats:
                    print(f"      - {emb_type}: {emb_stats['shape']}")

def preprocess_twitter_data(df, 
                          text_column='text',
                          config=None,
                          steps=['cleaning', 'tokenization', 'embeddings']):
    """
    Main function to preprocess Twitter data
    
    Args:
        df: Input DataFrame with raw Twitter data
        text_column: Name of the text column
        config: Configuration dictionary for preprocessing parameters
        steps: List of preprocessing steps to perform
        
    Returns:
        Dictionary containing all results and statistics
    """
    print("🚀 Starting Twitter data preprocessing...")
    
    if df.empty:
        print("   ❌ Empty DataFrame provided")
        return {}
    
    # Initialize pipeline
    pipeline = TwitterPreprocessingPipeline(config)
    
    # Run selected steps or complete pipeline
    if len(steps) == 3 and all(step in ['cleaning', 'tokenization', 'embeddings'] for step in steps):
        # Run complete pipeline
        results = pipeline.run_complete_pipeline(df, text_column)
    else:
        # Run individual steps
        print("   ℹ️ Running individual preprocessing steps...")
        current_df = df
        
        if 'cleaning' in steps:
            current_df = pipeline.step1_text_cleaning(current_df, text_column)
            text_column = 'text_cleaned'
        
        if 'tokenization' in steps and not current_df.empty:
            current_df = pipeline.step2_tokenization(current_df, text_column)
        
        if 'embeddings' in steps and not current_df.empty:
            pipeline.step3_embeddings_generation(current_df)
        
        results = pipeline._prepare_final_results()
    
    print("✅ Twitter data preprocessing completed")
    return results

if __name__ == "__main__":
    # Test the complete preprocessing pipeline
    print("🧪 Testing Twitter Preprocessing Pipeline...")
    
    # Sample data
    test_data = {
        'text': [
            "RT @user: Check out this amazing #Bitcoin news! 🚀 https://example.com",
            "I love #crypto and #bitcoin!!! 😍😍😍 @elonmusk what do you think???",
            "&lt;b&gt;Bitcoin&lt;/b&gt; is going to the moon! 🌙✨",
            "Just bought some ETH and BTC. HODL for the long term! #cryptocurrency #blockchain",
            "The crypto market is volatile but I believe in DeFi. It's the future of finance.",
            "Breaking: Bitcoin ETF approved by SEC! This is huge for crypto adoption. 📈",
            "FOMO is real in this bull market. Diamond hands only! 💎🙌",
            ""
        ],
        'date': ['2025-09-20'] * 8,
        'user': ['user1', 'user2', 'user3', 'user4', 'user5', 'user6', 'user7', 'user8']
    }
    
    df_test = pd.DataFrame(test_data)
    print(f"Test data shape: {df_test.shape}")
    
    # Test configuration
    test_config = {
        'embeddings': {
            'embedding_types': ['tfidf', 'word2vec'],  # Skip fasttext and sentence_bert for faster testing
            'save_embeddings': False
        },
        'output': {
            'save_intermediate': False,
            'save_final': False
        }
    }
    
    # Run preprocessing
    results = preprocess_twitter_data(df_test, config=test_config)
    
    print(f"\nPreprocessing completed!")
    print(f"Steps completed: {list(results['statistics'].keys())}")
    
    if 'tokenized_data' in results['data']:
        final_df = results['data']['tokenized_data']
        print(f"Final data shape: {final_df.shape}")
        print(f"Sample tokens: {final_df['tokens'].iloc[0] if len(final_df) > 0 else 'None'}")
