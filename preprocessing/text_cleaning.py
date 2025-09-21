#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text Cleaning Module for Bitcoin Price Prediction
Xử lý và làm sạch dữ liệu text từ Twitter

Author: Bitcoin Price Prediction Team
Date: 2025-09-20
"""

import re
import html
import pandas as pd
import numpy as np
from emoji_dict import EMOJI_DICT
import warnings
warnings.filterwarnings('ignore')

class TwitterTextCleaner:
    """Class để xử lý và làm sạch text Twitter"""
    
    def __init__(self):
        """Khởi tạo text cleaner với các regex patterns"""
        # URL patterns
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.short_url_pattern = re.compile(r'(?:^|\s)(?:www\.)?[a-zA-Z0-9-]+\.(?:com|org|net|edu|gov|mil|int|co|uk|de|fr|jp|cn)\S*')
        
        # Social media patterns
        self.mention_pattern = re.compile(r'@[A-Za-z0-9_]+')
        self.hashtag_pattern = re.compile(r'#[A-Za-z0-9_]+')
        
        # Special characters and noise
        self.special_chars = re.compile(r'[^\w\s#@]')
        self.multiple_spaces = re.compile(r'\s+')
        self.rt_pattern = re.compile(r'^RT\s+', re.IGNORECASE)
        
        # Load emoji dictionary
        self.emoji_dict = EMOJI_DICT
        
    def decode_html(self, text):
        """Decode HTML entities"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        return html.unescape(text)
    
    def convert_emojis(self, text):
        """Convert emojis to text descriptions"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        for emoji, description in self.emoji_dict.items():
            if emoji in text:
                text = text.replace(emoji, f" {description} ")
        return text
    
    def remove_urls(self, text):
        """Remove URLs from text"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Remove full URLs
        text = self.url_pattern.sub('', text)
        # Remove short URLs and domains
        text = self.short_url_pattern.sub('', text)
        return text
    
    def handle_mentions_hashtags(self, text, keep_hashtags=True, keep_mentions=False):
        """Handle mentions and hashtags"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        if not keep_mentions:
            text = self.mention_pattern.sub('', text)
        
        if not keep_hashtags:
            text = self.hashtag_pattern.sub('', text)
        else:
            # Remove # symbol but keep the word
            text = re.sub(r'#([A-Za-z0-9_]+)', r'\1', text)
        
        return text
    
    def remove_rt_prefix(self, text):
        """Remove RT (retweet) prefix"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        return self.rt_pattern.sub('', text)
    
    def normalize_text(self, text):
        """Normalize text - lowercase, remove extra spaces, etc."""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove repeated characters (keep max 2)
        text = self.repeated_chars.sub(r'\1\1', text)
        
        # Replace multiple spaces with single space
        text = self.multiple_spaces.sub(' ', text)
        
        # Strip whitespace
        text = text.strip()
        
        return text
    
    def remove_noise(self, text):
        """Remove various types of noise"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Remove special characters (keep basic punctuation)
        text = re.sub(r'[^\w\s.,!?;:\-\'\"#@]', ' ', text)
        
        # Remove extra punctuation
        text = re.sub(r'[.,!?;:]{2,}', '.', text)
        
        return text
    
    def clean_single_text(self, text, 
                         decode_html=True,
                         convert_emojis=True,
                         remove_urls=True,
                         handle_mentions=True,
                         handle_hashtags=True,
                         remove_rt=True,
                         remove_noise=True,
                         normalize=True,
                         keep_hashtags=True,
                         keep_mentions=False):
        """
        Clean a single text with all specified operations
        
        Args:
            text: Input text to clean
            decode_html: Whether to decode HTML entities
            convert_emojis: Whether to convert emojis to text
            remove_urls: Whether to remove URLs
            handle_mentions: Whether to process mentions
            handle_hashtags: Whether to process hashtags
            remove_rt: Whether to remove RT prefix
            remove_noise: Whether to remove noise
            normalize: Whether to normalize text
            keep_hashtags: Whether to keep hashtag content (remove # symbol only)
            keep_mentions: Whether to keep mention content
            
        Returns:
            Cleaned text string
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Step 1: HTML decoding
        if decode_html:
            text = self.decode_html(text)
        
        # Step 2: Emoji conversion
        if convert_emojis:
            text = self.convert_emojis(text)
        
        # Step 3: Remove RT prefix
        if remove_rt:
            text = self.remove_rt_prefix(text)
        
        # Step 4: URL removal
        if remove_urls:
            text = self.remove_urls(text)
        
        # Step 5: Handle mentions and hashtags
        if handle_mentions or handle_hashtags:
            text = self.handle_mentions_hashtags(text, 
                                               keep_hashtags=keep_hashtags,
                                               keep_mentions=keep_mentions)
        
        # Step 6: Remove noise
        if remove_noise:
            text = self.remove_noise(text)
        
        # Step 7: Normalization (always last)
        if normalize:
            text = self.normalize_text(text)
        
        return text
    
    def clean_dataframe(self, df, text_column='text', **kwargs):
        """
        Clean text data in a pandas DataFrame
        
        Args:
            df: Input DataFrame
            text_column: Name of the text column to clean
            **kwargs: Arguments passed to clean_single_text
            
        Returns:
            DataFrame with cleaned text and cleaning statistics
        """
        if df.empty or text_column not in df.columns:
            print(f"   ❌ Column '{text_column}' not found or DataFrame is empty")
            return df, {}
        
        print(f"   🧹 Cleaning {len(df)} texts in column '{text_column}'...")
        
        # Store original for comparison
        original_texts = df[text_column].copy()
        
        # Clean texts
        cleaned_texts = []
        for idx, text in enumerate(original_texts):
            cleaned = self.clean_single_text(text, **kwargs)
            cleaned_texts.append(cleaned)
            
            if (idx + 1) % 1000 == 0:
                print(f"      - Processed {idx + 1}/{len(original_texts)} texts")
        
        # Update DataFrame
        df_cleaned = df.copy()
        df_cleaned[f'{text_column}_cleaned'] = cleaned_texts
        
        # Calculate statistics
        stats = self._calculate_cleaning_stats(original_texts, cleaned_texts)
        
        # Filter out empty texts
        original_length = len(df_cleaned)
        df_cleaned = df_cleaned[df_cleaned[f'{text_column}_cleaned'].str.len() > 0]
        final_length = len(df_cleaned)
        
        stats['removed_empty_texts'] = original_length - final_length
        stats['final_text_count'] = final_length
        
        print(f"   ✅ Text cleaning completed")
        print(f"      - Original texts: {original_length}")
        print(f"      - Final texts: {final_length}")
        print(f"      - Removed empty: {stats['removed_empty_texts']}")
        print(f"      - Avg length reduction: {stats['avg_length_reduction']:.1f}%")
        
        return df_cleaned, stats
    
    def _calculate_cleaning_stats(self, original_texts, cleaned_texts):
        """Calculate cleaning statistics"""
        stats = {}
        
        # Length statistics
        original_lengths = [len(str(text)) if pd.notna(text) else 0 for text in original_texts]
        cleaned_lengths = [len(str(text)) if pd.notna(text) else 0 for text in cleaned_texts]
        
        stats['original_avg_length'] = np.mean(original_lengths)
        stats['cleaned_avg_length'] = np.mean(cleaned_lengths)
        stats['avg_length_reduction'] = ((stats['original_avg_length'] - stats['cleaned_avg_length']) / stats['original_avg_length']) * 100
        
        # Count various elements
        stats['urls_removed'] = sum(1 for text in original_texts if pd.notna(text) and ('http' in str(text) or 'www.' in str(text)))
        stats['mentions_found'] = sum(1 for text in original_texts if pd.notna(text) and '@' in str(text))
        stats['hashtags_found'] = sum(1 for text in original_texts if pd.notna(text) and '#' in str(text))
        stats['rt_tweets'] = sum(1 for text in original_texts if pd.notna(text) and str(text).lower().startswith('rt '))
        
        return stats

def clean_twitter_data(df, text_column='text', **cleaning_options):
    """
    Main function to clean Twitter data
    
    Args:
        df: Input DataFrame with Twitter data
        text_column: Name of the text column to clean
        **cleaning_options: Options for text cleaning
        
    Returns:
        Tuple of (cleaned_dataframe, cleaning_statistics)
    """
    print("🧹 Starting Twitter text cleaning...")
    
    if df.empty:
        print("   ❌ Empty DataFrame provided")
        return df, {}
    
    # Initialize cleaner
    cleaner = TwitterTextCleaner()
    
    # Default cleaning options
    default_options = {
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
    }
    
    # Update with user options
    default_options.update(cleaning_options)
    
    # Clean the data
    cleaned_df, stats = cleaner.clean_dataframe(df, text_column, **default_options)
    
    print("✅ Twitter text cleaning completed")
    return cleaned_df, stats

if __name__ == "__main__":
    # Test the cleaning functions
    print("🧪 Testing Twitter Text Cleaner...")
    
    # Sample data
    test_data = {
        'text': [
            "RT @user: Check out this amazing #Bitcoin news! 🚀 https://example.com",
            "I love #crypto and #bitcoin!!! 😍😍😍 @elonmusk what do you think???",
            "&lt;b&gt;Bitcoin&lt;/b&gt; is going to the moon! 🌙✨",
            "Visit www.example.com for more crypto news #BTC #ETH",
            ""
        ]
    }
    
    df_test = pd.DataFrame(test_data)
    print(f"Original data:\n{df_test}")
    
    cleaned_df, stats = clean_twitter_data(df_test)
    print(f"\nCleaned data:\n{cleaned_df[['text', 'text_cleaned']]}")
    print(f"\nCleaning statistics:\n{stats}")
