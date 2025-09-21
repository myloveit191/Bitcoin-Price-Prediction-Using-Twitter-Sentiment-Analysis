#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sentence Segmentation Module for Bitcoin Price Prediction
Phân đoạn câu cho dữ liệu Twitter

Author: Bitcoin Price Prediction Team
Date: 2025-09-20
"""

import re
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, PunktSentenceTokenizer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("📥 Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)

class TwitterSentenceSegmenter:
    """Class để phân đoạn câu cho Twitter data"""
    
    def __init__(self):
        """Khởi tạo sentence segmenter với các patterns Twitter-specific"""
        # Twitter-specific patterns
        self.tweet_end_patterns = [
            r'\.{1,3}\s*$',  # Ending with dots
            r'[!?]+\s*$',    # Ending with exclamation or question marks
            r'[.!?]\s*[#@]', # Sentence break before hashtag or mention
            r'\n+',          # Line breaks
        ]
        
        # Compile patterns
        self.compiled_patterns = [re.compile(pattern) for pattern in self.tweet_end_patterns]
        
        # Abbreviations common in crypto/finance tweets
        self.crypto_abbreviations = {
            'btc', 'eth', 'bnb', 'ada', 'xrp', 'dot', 'ltc', 'bch', 'link',
            'xlm', 'vet', 'trx', 'eos', 'xtz', 'atom', 'neo', 'iota', 'mkr',
            'usd', 'eur', 'gbp', 'jpy', 'cad', 'aud', 'chf', 'cny',
            'defi', 'nft', 'dao', 'dex', 'cex', 'ico', 'ipo', 'etf',
            'fomo', 'hodl', 'rekt', 'moon', 'bear', 'bull', 'whale'
        }
        
        # Initialize custom tokenizer
        self.punkt_tokenizer = PunktSentenceTokenizer()
        self._train_custom_tokenizer()
    
    def _train_custom_tokenizer(self):
        """Train tokenizer with crypto-specific abbreviations"""
        # Add crypto abbreviations as known abbreviations
        for abbrev in self.crypto_abbreviations:
            self.punkt_tokenizer._params.abbrev_types.add(abbrev.lower())
            self.punkt_tokenizer._params.abbrev_types.add(abbrev.upper())
    
    def segment_single_text(self, text, method='hybrid'):
        """
        Segment a single text into sentences
        
        Args:
            text: Input text to segment
            method: Segmentation method ('nltk', 'rule_based', 'hybrid')
            
        Returns:
            List of sentences
        """
        if pd.isna(text) or not isinstance(text, str) or len(text.strip()) == 0:
            return []
        
        text = text.strip()
        
        if method == 'nltk':
            return self._nltk_segmentation(text)
        elif method == 'rule_based':
            return self._rule_based_segmentation(text)
        elif method == 'hybrid':
            return self._hybrid_segmentation(text)
        else:
            raise ValueError(f"Unknown segmentation method: {method}")
    
    def _nltk_segmentation(self, text):
        """NLTK-based sentence segmentation"""
        try:
            sentences = self.punkt_tokenizer.tokenize(text)
            # Filter out very short sentences (likely noise)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
            return sentences
        except Exception as e:
            print(f"   ⚠️ NLTK segmentation failed: {e}")
            return [text]  # Return original text as single sentence
    
    def _rule_based_segmentation(self, text):
        """Rule-based segmentation for Twitter-specific structures"""
        sentences = []
        current_sentence = ""
        
        # Split by strong sentence boundaries
        parts = re.split(r'[.!?]+\s+', text)
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            # Check if this part should be combined with previous
            if self._should_combine_with_previous(current_sentence, part):
                current_sentence += " " + part
            else:
                if current_sentence:
                    sentences.append(current_sentence.strip())
                current_sentence = part
        
        # Add the last sentence
        if current_sentence:
            sentences.append(current_sentence.strip())
        
        # Filter short sentences
        sentences = [s for s in sentences if len(s.strip()) > 3]
        return sentences if sentences else [text]
    
    def _hybrid_segmentation(self, text):
        """Hybrid approach combining NLTK and rule-based methods"""
        # Start with NLTK
        nltk_sentences = self._nltk_segmentation(text)
        
        # If NLTK produces only one sentence, try rule-based
        if len(nltk_sentences) <= 1:
            rule_sentences = self._rule_based_segmentation(text)
            if len(rule_sentences) > len(nltk_sentences):
                return rule_sentences
        
        # Post-process NLTK results with Twitter-specific rules
        processed_sentences = []
        for sentence in nltk_sentences:
            # Split further if contains Twitter-specific patterns
            sub_sentences = self._split_twitter_patterns(sentence)
            processed_sentences.extend(sub_sentences)
        
        return processed_sentences if processed_sentences else [text]
    
    def _should_combine_with_previous(self, previous, current):
        """Check if current part should be combined with previous sentence"""
        if not previous:
            return False
        
        # Don't combine if current starts with capital letter (likely new sentence)
        if current and current[0].isupper():
            return False
        
        # Don't combine if previous ends with strong punctuation
        if previous.rstrip().endswith(('.', '!', '?')):
            return False
        
        # Combine if current is very short (likely continuation)
        if len(current.split()) <= 2:
            return True
        
        return False
    
    def _split_twitter_patterns(self, sentence):
        """Split sentence based on Twitter-specific patterns"""
        sentences = [sentence]
        
        # Split on hashtag clusters (multiple hashtags together)
        hashtag_pattern = r'(\s+#\w+(?:\s+#\w+){2,})'
        for i, sent in enumerate(sentences):
            if re.search(hashtag_pattern, sent):
                parts = re.split(hashtag_pattern, sent)
                sentences[i:i+1] = [p.strip() for p in parts if p.strip()]
        
        # Split very long sentences (likely multiple thoughts)
        max_length = 200  # characters
        long_sentences = []
        for sent in sentences:
            if len(sent) > max_length:
                # Try to split at natural break points
                break_points = ['. ', '! ', '? ', ', and ', ', but ', ', so ']
                best_split = None
                best_position = 0
                
                for bp in break_points:
                    pos = sent.find(bp, max_length // 2)
                    if pos > 0 and pos < max_length:
                        if pos > best_position:
                            best_position = pos
                            best_split = bp
                
                if best_split:
                    parts = sent.split(best_split, 1)
                    long_sentences.extend([parts[0] + best_split.rstrip(), parts[1].strip()])
                else:
                    long_sentences.append(sent)
            else:
                long_sentences.append(sent)
        
        return [s for s in long_sentences if s.strip()]
    
    def segment_dataframe(self, df, text_column='text_cleaned', method='hybrid'):
        """
        Segment sentences in a pandas DataFrame
        
        Args:
            df: Input DataFrame
            text_column: Name of the text column to segment
            method: Segmentation method
            
        Returns:
            DataFrame with segmented sentences and statistics
        """
        if df.empty or text_column not in df.columns:
            print(f"   ❌ Column '{text_column}' not found or DataFrame is empty")
            return df, {}
        
        print(f"   ✂️ Segmenting sentences in {len(df)} texts using '{method}' method...")
        
        segmented_data = []
        total_sentences = 0
        
        for idx, row in df.iterrows():
            text = row[text_column]
            sentences = self.segment_single_text(text, method=method)
            
            for sent_idx, sentence in enumerate(sentences):
                new_row = row.copy()
                new_row['sentence'] = sentence
                new_row['sentence_id'] = f"{idx}_{sent_idx}"
                new_row['original_text_id'] = idx
                new_row['sentence_index'] = sent_idx
                new_row['total_sentences_in_text'] = len(sentences)
                segmented_data.append(new_row)
            
            total_sentences += len(sentences)
            
            if (idx + 1) % 1000 == 0:
                print(f"      - Processed {idx + 1}/{len(df)} texts")
        
        # Create new DataFrame
        df_segmented = pd.DataFrame(segmented_data)
        
        # Calculate statistics
        stats = {
            'original_texts': len(df),
            'total_sentences': total_sentences,
            'avg_sentences_per_text': total_sentences / len(df) if len(df) > 0 else 0,
            'method_used': method
        }
        
        # Sentence length statistics
        if not df_segmented.empty:
            sentence_lengths = df_segmented['sentence'].str.len()
            stats['avg_sentence_length'] = sentence_lengths.mean()
            stats['median_sentence_length'] = sentence_lengths.median()
            stats['min_sentence_length'] = sentence_lengths.min()
            stats['max_sentence_length'] = sentence_lengths.max()
        
        print(f"   ✅ Sentence segmentation completed")
        print(f"      - Original texts: {stats['original_texts']}")
        print(f"      - Total sentences: {stats['total_sentences']}")
        print(f"      - Avg sentences per text: {stats['avg_sentences_per_text']:.1f}")
        print(f"      - Avg sentence length: {stats['avg_sentence_length']:.1f} chars")
        
        return df_segmented, stats

def segment_twitter_sentences(df, text_column='text_cleaned', method='hybrid'):
    """
    Main function to segment Twitter sentences
    
    Args:
        df: Input DataFrame with Twitter data
        text_column: Name of the text column to segment
        method: Segmentation method ('nltk', 'rule_based', 'hybrid')
        
    Returns:
        Tuple of (segmented_dataframe, segmentation_statistics)
    """
    print("✂️ Starting Twitter sentence segmentation...")
    
    if df.empty:
        print("   ❌ Empty DataFrame provided")
        return df, {}
    
    # Initialize segmenter
    segmenter = TwitterSentenceSegmenter()
    
    # Segment sentences
    segmented_df, stats = segmenter.segment_dataframe(df, text_column, method)
    
    print("✅ Twitter sentence segmentation completed")
    return segmented_df, stats

if __name__ == "__main__":
    # Test the segmentation functions
    print("🧪 Testing Twitter Sentence Segmenter...")
    
    # Sample data
    test_data = {
        'text_cleaned': [
            "bitcoin is going up today. i think it will reach new heights! what do you think about btc price?",
            "just bought some eth and btc hodling for the long term",
            "crypto market is volatile but i believe in blockchain technology defi is the future",
            "breaking news bitcoin etf approved by sec this is huge for crypto adoption",
            ""
        ]
    }
    
    df_test = pd.DataFrame(test_data)
    print(f"Original data:\n{df_test}")
    
    segmented_df, stats = segment_twitter_sentences(df_test)
    print(f"\nSegmented data:\n{segmented_df[['text_cleaned', 'sentence', 'sentence_index']]}")
    print(f"\nSegmentation statistics:\n{stats}") 