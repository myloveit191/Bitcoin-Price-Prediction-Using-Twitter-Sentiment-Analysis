#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tokenization Module for Bitcoin Price Prediction
Advanced tokenization using WordPiece and SentencePiece for Twitter sentences
Focus: Subword tokenization with crypto vocabulary preservation

Author: Bitcoin Price Prediction Team
Date: 2025-09-20
"""

import re
import os
import pandas as pd
import numpy as np
import sentencepiece as spm
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, normalizers
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.normalizers import NFD, Lowercase, StripAccents
import warnings
from nltk.tokenize import TweetTokenizer
warnings.filterwarnings('ignore')

class TwitterTokenizer:
    """Class để tokenize cleaned Twitter sentences với WordPiece/SentencePiece và crypto vocabulary preservation"""
    
    def __init__(self, tokenizer_type='sentencepiece', vocab_size=8000, model_path=None):
        """
        Khởi tạo tokenizer với advanced tokenization techniques
        
        Args:
            tokenizer_type: 'sentencepiece', 'wordpiece', hoặc 'hybrid'
            vocab_size: Kích thước vocabulary cho subword tokenization
            model_path: Đường dẫn đến model đã train (nếu có)
        """
        self.tokenizer_type = tokenizer_type
        self.vocab_size = vocab_size
        self.model_path = model_path
        
        # Initialize basic Twitter tokenizer for preprocessing
        self.tweet_tokenizer = TweetTokenizer(
            preserve_case=False,
            reduce_len=True,
            strip_handles=False
        )
        
        # Crypto vocabulary to preserve (được thêm vào special tokens)
        self.crypto_vocabulary = {
            # Cryptocurrencies
            'bitcoin', 'btc', 'ethereum', 'eth', 'binance', 'bnb', 'cardano', 'ada',
            'ripple', 'xrp', 'polkadot', 'dot', 'litecoin', 'ltc', 'chainlink', 'link',
            'stellar', 'xlm', 'vechain', 'vet', 'tron', 'trx', 'eos', 'tezos', 'xtz',
            'cosmos', 'atom', 'neo', 'iota', 'maker', 'mkr', 'dogecoin', 'doge',
            'shiba', 'shib', 'avalanche', 'avax', 'solana', 'sol', 'polygon', 'matic',
            
            # Fiat currencies
            'usd', 'eur', 'gbp', 'jpy', 'cad', 'aud', 'chf', 'cny', 'krw', 'inr',
            'dollar', 'euro', 'pound', 'yen', 'yuan', 'won', 'rupee',
            
            # Crypto terms
            'defi', 'nft', 'dao', 'dex', 'cex', 'ico', 'ipo', 'etf', 'stablecoin',
            'blockchain', 'cryptocurrency', 'altcoin', 'memecoin', 'token', 'coin',
            'wallet', 'mining', 'staking', 'yield', 'farming', 'liquidity', 'swap',
            'bridge', 'oracle', 'smart', 'contract', 'consensus', 'node', 'validator',
            
            # Trading terms
            'hodl', 'fomo', 'fud', 'rekt', 'moon', 'lambo', 'diamond', 'hands',
            'paper', 'whale', 'bull', 'bear', 'pump', 'dump', 'ath', 'atl',
            'market', 'cap', 'volume', 'price', 'support', 'resistance', 'breakout',
            'correction', 'crash', 'rally', 'trend', 'analysis', 'ta', 'fa',
            
            # Exchanges
            'binance', 'coinbase', 'kraken', 'bitfinex', 'huobi', 'okx', 'kucoin',
            'bybit', 'ftx', 'gemini', 'bitstamp', 'pancakeswap', 'uniswap', 'sushiswap'
        }
        
        # Initialize tokenizers
        self.sentencepiece_model = None
        self.wordpiece_tokenizer = None
        self.hybrid_tokenizer = None
        
        # Load existing model if provided
        if model_path and os.path.exists(os.path.dirname(model_path)):
            self.load_model(model_path)
        elif model_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    def _calculate_adaptive_vocab_size(self, texts, base_vocab_size):
        """Calculate adaptive vocabulary size based on data size"""
        if not texts:
            return min(base_vocab_size, 100)
        
        # Count unique characters and words
        all_text = ' '.join(texts)
        unique_chars = len(set(all_text.lower()))
        unique_words = len(set(all_text.lower().split()))
        
        # Calculate reasonable vocab size
        # Base on unique words + crypto vocabulary + some buffer
        crypto_vocab_size = len(self.crypto_vocabulary)
        adaptive_size = min(
            base_vocab_size,
            max(100, unique_words + crypto_vocab_size + 50)
        )
        
        print(f"   📊 Adaptive vocab size: {adaptive_size} (unique words: {unique_words}, crypto terms: {crypto_vocab_size})")
        return adaptive_size
    
    def _create_sentencepiece_model(self, texts, model_prefix='crypto_twitter_sp'):
        """Tạo SentencePiece model từ training data"""
        # Calculate adaptive vocabulary size
        adaptive_vocab_size = self._calculate_adaptive_vocab_size(texts, self.vocab_size)
        
        print(f"   🔧 Training SentencePiece model with vocab_size={adaptive_vocab_size}...")
        
        # Use model_path if available, otherwise use model_prefix
        if self.model_path:
            model_prefix = self.model_path
        
        # Prepare training data
        temp_file = f"{model_prefix}_temp.txt"
        with open(temp_file, 'w', encoding='utf-8') as f:
            for text in texts:
                if isinstance(text, str) and len(text.strip()) > 0:
                    f.write(text.strip() + '\n')
        
        # Create user defined symbols for crypto vocabulary (limit to avoid issues)
        user_defined_symbols = list(self.crypto_vocabulary)[:50]  # Limit to first 50
        
        try:
            # Train SentencePiece model
            spm.SentencePieceTrainer.train(
                input=temp_file,
                model_prefix=model_prefix,
                vocab_size=adaptive_vocab_size,
                character_coverage=0.995,
                model_type='bpe',  # Byte-pair encoding
                user_defined_symbols=user_defined_symbols,
                pad_id=0,
                unk_id=1,
                bos_id=2,
                eos_id=3,
                normalization_rule_name='nmt_nfkc_cf',
                max_sentence_length=4192,
                shuffle_input_sentence=True
            )
            
            # Load the trained model
            sp_model = spm.SentencePieceProcessor()
            sp_model.load(f"{model_prefix}.model")
            
            print(f"   ✅ SentencePiece model trained successfully")
            return sp_model
            
        except Exception as e:
            print(f"   ❌ Error training SentencePiece model: {e}")
            print(f"   🔄 Falling back to basic tokenization...")
            return None
        finally:
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def _create_wordpiece_tokenizer(self, texts):
        """Tạo WordPiece tokenizer từ training data"""
        # Calculate adaptive vocabulary size
        adaptive_vocab_size = self._calculate_adaptive_vocab_size(texts, self.vocab_size)
        
        print(f"   �� Training WordPiece tokenizer with vocab_size={adaptive_vocab_size}...")
        
        try:
            # Initialize WordPiece tokenizer
            tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
            
            # Add normalizers
            tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
            
            # Add pre-tokenizer
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            
            # Prepare special tokens including crypto vocabulary (limit to avoid issues)
            special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
            special_tokens.extend(list(self.crypto_vocabulary)[:50])  # Limit to first 50
            
            # Initialize trainer
            trainer = WordPieceTrainer(
                vocab_size=adaptive_vocab_size,
                special_tokens=special_tokens,
                min_frequency=1  # Allow single occurrence
            )
            
            # Train tokenizer
            tokenizer.train_from_iterator(texts, trainer)
            
            print(f"   ✅ WordPiece tokenizer trained successfully")
            return tokenizer
            
        except Exception as e:
            print(f"   ❌ Error training WordPiece tokenizer: {e}")
            print(f"   🔄 Falling back to basic tokenization...")
            return None
    
    def _create_hybrid_tokenizer(self, texts):
        """Tạo hybrid tokenizer kết hợp WordPiece và SentencePiece"""
        print(f"   🔧 Creating hybrid tokenizer...")
        
        try:
            # Create both models
            sp_model = self._create_sentencepiece_model(texts, 'hybrid_sp')
            wp_tokenizer = self._create_wordpiece_tokenizer(texts)
            
            if sp_model and wp_tokenizer:
                print(f"   ✅ Hybrid tokenizer created successfully")
                return {'sentencepiece': sp_model, 'wordpiece': wp_tokenizer}
            else:
                print(f"   ⚠️ Hybrid tokenizer creation failed, using fallback")
                return None
                
        except Exception as e:
            print(f"   ❌ Error creating hybrid tokenizer: {e}")
            return None
    
    def train_tokenizer(self, texts):
        """
        Train tokenizer trên dữ liệu texts
        
        Args:
            texts: List of texts để train tokenizer
        """
        print(f"🔧 Training {self.tokenizer_type} tokenizer...")
        
        if not texts or len(texts) == 0:
            print("   ❌ No training data provided")
            return
        
        # Filter valid texts
        valid_texts = [text for text in texts if isinstance(text, str) and len(text.strip()) > 0]
        print(f"   📊 Training on {len(valid_texts)} valid texts")
        
        if len(valid_texts) < 5:
            print("   ⚠️ Very small dataset, using basic tokenization")
            return
        
        try:
            if self.tokenizer_type == 'sentencepiece':
                self.sentencepiece_model = self._create_sentencepiece_model(valid_texts)
            elif self.tokenizer_type == 'wordpiece':
                self.wordpiece_tokenizer = self._create_wordpiece_tokenizer(valid_texts)
            elif self.tokenizer_type == 'hybrid':
                self.hybrid_tokenizer = self._create_hybrid_tokenizer(valid_texts)
            
            print("✅ Tokenizer training completed")
            
        except Exception as e:
            print(f"   ❌ Error during training: {e}")
            print("    Will use basic tokenization as fallback")
    
    def tokenize_single_text(self, text, 
                           max_length=512,
                           add_special_tokens=True,
                           return_tokens=True,
                           return_ids=False):
        """
        Tokenize một text đơn lẻ sử dụng trained tokenizer
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            add_special_tokens: Whether to add special tokens
            return_tokens: Whether to return tokens
            return_ids: Whether to return token IDs
            
        Returns:
            Dict containing tokens and/or IDs
        """
        if pd.isna(text) or not isinstance(text, str) or len(text.strip()) == 0:
            return {'tokens': [], 'ids': []} if return_ids else []
        
        text = text.strip().lower()
        result = {}
        
        try:
            if self.tokenizer_type == 'sentencepiece' and self.sentencepiece_model:
                # SentencePiece tokenization
                if return_tokens:
                    tokens = self.sentencepiece_model.encode_as_pieces(text)
                    if max_length and len(tokens) > max_length:
                        tokens = tokens[:max_length]
                    result['tokens'] = tokens
                
                if return_ids:
                    ids = self.sentencepiece_model.encode_as_ids(text)
                    if max_length and len(ids) > max_length:
                        ids = ids[:max_length]
                    result['ids'] = ids
                    
            elif self.tokenizer_type == 'wordpiece' and self.wordpiece_tokenizer:
                # WordPiece tokenization
                encoding = self.wordpiece_tokenizer.encode(text, add_special_tokens=add_special_tokens)
                
                if return_tokens:
                    tokens = encoding.tokens
                    if max_length and len(tokens) > max_length:
                        tokens = tokens[:max_length]
                    result['tokens'] = tokens
                
                if return_ids:
                    ids = encoding.ids
                    if max_length and len(ids) > max_length:
                        ids = ids[:max_length]
                    result['ids'] = ids
                    
            elif self.tokenizer_type == 'hybrid' and self.hybrid_tokenizer:
                # Hybrid tokenization - use both and combine results
                sp_tokens = self.hybrid_tokenizer['sentencepiece'].encode_as_pieces(text)
                wp_encoding = self.hybrid_tokenizer['wordpiece'].encode(text, add_special_tokens=add_special_tokens)
                
                # Combine tokens (prioritize SentencePiece for crypto terms)
                combined_tokens = []
                crypto_found = any(crypto_word in text.lower() for crypto_word in self.crypto_vocabulary)
                
                if crypto_found:
                    # Use SentencePiece for crypto-heavy content
                    combined_tokens = sp_tokens
                else:
                    # Use WordPiece for general content
                    combined_tokens = wp_encoding.tokens
                
                if max_length and len(combined_tokens) > max_length:
                    combined_tokens = combined_tokens[:max_length]
                
                result['tokens'] = combined_tokens if return_tokens else []
                result['ids'] = wp_encoding.ids[:max_length] if return_ids else []
            
            else:
                # Fallback to basic tokenization
                basic_tokens = self.tweet_tokenizer.tokenize(text)
                if max_length and len(basic_tokens) > max_length:
                    basic_tokens = basic_tokens[:max_length]
                result['tokens'] = basic_tokens if return_tokens else []
                result['ids'] = list(range(len(basic_tokens))) if return_ids else []
                
        except Exception as e:
            print(f"   ⚠️ Error in tokenization: {e}, using fallback")
            # Fallback to basic tokenization
            basic_tokens = self.tweet_tokenizer.tokenize(text)
            if max_length and len(basic_tokens) > max_length:
                basic_tokens = basic_tokens[:max_length]
            result['tokens'] = basic_tokens if return_tokens else []
            result['ids'] = list(range(len(basic_tokens))) if return_ids else []
        
        # Return format based on requirements
        if return_tokens and return_ids:
            return result
        elif return_tokens:
            return result.get('tokens', [])
        elif return_ids:
            return result.get('ids', [])
        else:
            return result.get('tokens', [])
    
    def tokenize_dataframe(self, df, text_column='sentence', **kwargs):
        """
        Tokenize text data trong pandas DataFrame
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
            **kwargs: Arguments for tokenize_single_text
            
        Returns:
            DataFrame with tokenized text and statistics
        """
        if df.empty or text_column not in df.columns:
            print(f"   ❌ Column '{text_column}' not found or DataFrame is empty")
            return df, {}
        
        print(f"   🔤 Tokenizing {len(df)} texts using {self.tokenizer_type} tokenizer...")
        
        # Check if tokenizer is trained
        if (self.tokenizer_type == 'sentencepiece' and not self.sentencepiece_model) or \
           (self.tokenizer_type == 'wordpiece' and not self.wordpiece_tokenizer) or \
           (self.tokenizer_type == 'hybrid' and not self.hybrid_tokenizer):
            print("   🔧 Tokenizer not trained. Training on current data...")
            self.train_tokenizer(df[text_column].tolist())
        
        tokenized_data = []
        total_tokens = 0
        
        for idx, row in df.iterrows():
            text = row[text_column]
            
            # Get tokenization result
            token_result = self.tokenize_single_text(text, **kwargs)
            
            if isinstance(token_result, dict):
                tokens = token_result.get('tokens', [])
                token_ids = token_result.get('ids', [])
            else:
                tokens = token_result
                token_ids = []
            
            # Create new row
            new_row = row.copy()
            new_row['tokens'] = tokens
            new_row['token_ids'] = token_ids
            new_row['token_count'] = len(tokens)
            new_row['tokens_string'] = ' '.join(tokens) if tokens else ''
            tokenized_data.append(new_row)
            
            total_tokens += len(tokens)
            
            if (idx + 1) % 1000 == 0:
                print(f"      - Processed {idx + 1}/{len(df)} texts")
        
        # Create DataFrame
        df_tokenized = pd.DataFrame(tokenized_data)
        
        # Filter empty tokenizations
        original_length = len(df_tokenized)
        df_tokenized = df_tokenized[df_tokenized['token_count'] > 0]
        final_length = len(df_tokenized)
        
        # Calculate statistics
        stats = self._calculate_tokenization_stats(df_tokenized, total_tokens, 
                                                 original_length, final_length)
        
        print(f"   ✅ Tokenization completed")
        print(f"      - Tokenizer type: {self.tokenizer_type}")
        print(f"      - Original texts: {original_length}")
        print(f"      - Final texts: {final_length}")
        print(f"      - Total tokens: {total_tokens}")
        print(f"      - Avg tokens per text: {stats['avg_tokens_per_text']:.1f}")
        
        return df_tokenized, stats
    
    def _calculate_tokenization_stats(self, df, total_tokens, original_length, final_length):
        """Calculate tokenization statistics"""
        stats = {
            'tokenizer_type': self.tokenizer_type,
            'vocab_size': self.vocab_size,
            'original_texts': original_length,
            'final_texts': final_length,
            'removed_empty_texts': original_length - final_length,
            'total_tokens': total_tokens,
            'avg_tokens_per_text': total_tokens / final_length if final_length > 0 else 0
        }
        
        if not df.empty:
            token_counts = df['token_count']
            stats['median_tokens_per_text'] = token_counts.median()
            stats['min_tokens_per_text'] = token_counts.min()
            stats['max_tokens_per_text'] = token_counts.max()
            stats['std_tokens_per_text'] = token_counts.std()
            
            # Vocabulary statistics
            all_tokens = []
            for tokens in df['tokens']:
                if isinstance(tokens, list):
                    all_tokens.extend(tokens)
            
            stats['unique_tokens'] = len(set(all_tokens))
            stats['vocabulary_size'] = len(set(all_tokens))
            
            # Most common tokens
            from collections import Counter
            token_counter = Counter(all_tokens)
            stats['most_common_tokens'] = token_counter.most_common(10)
            
            # Crypto vocabulary usage
            crypto_tokens = [token for token in all_tokens 
                           if any(crypto_word in token.lower() for crypto_word in self.crypto_vocabulary)]
            stats['crypto_token_count'] = len(crypto_tokens)
            stats['crypto_token_ratio'] = len(crypto_tokens) / len(all_tokens) if all_tokens else 0
            stats['unique_crypto_tokens'] = len(set(crypto_tokens))
        
        return stats
    
    def save_model(self, save_path):
        """Lưu trained tokenizer model"""
        if not save_path:
            print("   ❌ No save path provided")
            return
            
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        try:
            if self.tokenizer_type == 'sentencepiece' and self.sentencepiece_model:
                # SentencePiece model files are already saved during training
                # Just verify they exist
                if os.path.exists(f"{save_path}.model"):
                    print(f"   💾 SentencePiece model saved at {save_path}")
                else:
                    print(f"   ❌ SentencePiece model file not found at {save_path}.model")
            elif self.tokenizer_type == 'wordpiece' and self.wordpiece_tokenizer:
                self.wordpiece_tokenizer.save(save_path)
                print(f"   💾 WordPiece tokenizer saved at {save_path}")
            elif self.tokenizer_type == 'hybrid' and self.hybrid_tokenizer:
                # Save both models
                self.hybrid_tokenizer['wordpiece'].save(f"{save_path}_wp.json")
                print(f"   💾 Hybrid tokenizer saved at {save_path}")
        except Exception as e:
            print(f"   ❌ Error saving model: {e}")
    
    def load_model(self, model_path):
        """Load trained tokenizer model"""
        if not model_path:
            print("   ⚠️ No model path provided")
            return
            
        try:
            if self.tokenizer_type == 'sentencepiece':
                model_file = f"{model_path}.model"
                if os.path.exists(model_file):
                    self.sentencepiece_model = spm.SentencePieceProcessor()
                    self.sentencepiece_model.load(model_file)
                    print(f"   📂 SentencePiece model loaded from {model_path}")
                else:
                    print(f"   ⚠️ SentencePiece model file not found: {model_file}")
                
            elif self.tokenizer_type == 'wordpiece':
                if os.path.exists(model_path):
                    self.wordpiece_tokenizer = Tokenizer.from_file(model_path)
                    print(f"   📂 WordPiece tokenizer loaded from {model_path}")
                else:
                    print(f"   ⚠️ WordPiece model file not found: {model_path}")
                    
            elif self.tokenizer_type == 'hybrid':
                wp_path = f"{model_path}_wp.json"
                sp_path = f"{model_path}.model"
                
                if os.path.exists(wp_path) and os.path.exists(sp_path):
                    wp_tokenizer = Tokenizer.from_file(wp_path)
                    sp_model = spm.SentencePieceProcessor()
                    sp_model.load(sp_path)
                    self.hybrid_tokenizer = {'wordpiece': wp_tokenizer, 'sentencepiece': sp_model}
                    print(f"   📂 Hybrid tokenizer loaded from {model_path}")
                else:
                    print(f"   ⚠️ Hybrid model files not found: {wp_path}, {sp_path}")
                    
        except Exception as e:
            print(f"   ❌ Error loading model: {e}")
    
    def create_vocabulary(self, df, min_frequency=2, max_vocab_size=10000):
        """
        Create vocabulary từ tokenized DataFrame
        
        Args:
            df: DataFrame with tokenized text
            min_frequency: Minimum frequency for inclusion
            max_vocab_size: Maximum vocabulary size
            
        Returns:
            Dictionary mapping tokens to indices
        """
        if df.empty or 'tokens' not in df.columns:
            print("   ❌ No tokenized data found")
            return {}
        
        print(f"   📚 Creating vocabulary from {len(df)} texts...")
        
        # Count token frequencies
        from collections import Counter
        all_tokens = []
        for tokens in df['tokens']:
            if isinstance(tokens, list):
                all_tokens.extend(tokens)
        
        token_counter = Counter(all_tokens)
        
        # Filter by frequency
        filtered_tokens = [token for token, freq in token_counter.items() 
                          if freq >= min_frequency]
        
        # Sort by frequency and limit size
        sorted_tokens = sorted(filtered_tokens, 
                             key=lambda x: token_counter[x], 
                             reverse=True)[:max_vocab_size]
        
        # Create vocabulary mapping
        vocab = {'<PAD>': 0, '<UNK>': 1, '<CLS>': 2, '<SEP>': 3}  # Special tokens
        for i, token in enumerate(sorted_tokens):
            vocab[token] = i + 4
        
        print(f"   ✅ Vocabulary created")
        print(f"      - Total unique tokens: {len(token_counter)}")
        print(f"      - Filtered tokens (freq >= {min_frequency}): {len(filtered_tokens)}")
        print(f"      - Final vocabulary size: {len(vocab)}")
        
        return vocab

def tokenize_twitter_data(df, text_column='sentence', tokenizer_type='sentencepiece', 
                         vocab_size=8000, model_path=None, **tokenization_options):
    """
    Main function để tokenize Twitter data với advanced techniques
    
    Args:
        df: Input DataFrame
        text_column: Name of text column
        tokenizer_type: 'sentencepiece', 'wordpiece', hoặc 'hybrid'
        vocab_size: Vocabulary size cho subword tokenization
        model_path: Path to pretrained model (optional)
        **tokenization_options: Additional options
        
    Returns:
        Tuple of (tokenized_dataframe, tokenization_statistics)
    """
    print(f"🔤 Starting advanced Twitter tokenization with {tokenizer_type}...")
    
    if df.empty:
        print("   ❌ Empty DataFrame provided")
        return df, {}
    
    # Initialize tokenizer
    tokenizer = TwitterTokenizer(
        tokenizer_type=tokenizer_type,
        vocab_size=vocab_size,
        model_path=model_path
    )
    
    # Default options
    default_options = {
        'max_length': 512,
        'add_special_tokens': True,
        'return_tokens': True,
        'return_ids': False
    }
    
    # Update with user options
    default_options.update(tokenization_options)
    
    # Tokenize data
    tokenized_df, stats = tokenizer.tokenize_dataframe(df, text_column, **default_options)
    
    print(f"✅ Advanced Twitter tokenization completed with {tokenizer_type}")
    return tokenized_df, stats

if __name__ == "__main__":
    # Test advanced tokenization
    print("🧪 Testing Advanced Crypto-Aware Twitter Tokenizer...")
    
    # Sample data
    test_data = {
        'sentence': [
            "bitcoin is going to the moon with diamond hands hodl",
            "i think btc price will crash very badly soon",
            "ethereum defi protocols are absolutely amazing right now",
            "this bearish trend is terrible and i am scared",
            "blockchain technology and cryptocurrency adoption growing",
            "fomo buying more altcoins thinking about lambo dreams",
            "market analysis shows bullish sentiment for crypto",
            "staking rewards in defi yield farming protocols",
            "nft marketplace on polygon network very exciting",
            "trading volume increasing on major exchanges today"
        ]
    }
    
    df_test = pd.DataFrame(test_data)
    
    # Test different tokenizer types with adaptive vocab size
    for tokenizer_type in ['sentencepiece', 'wordpiece', 'hybrid']:
        print(f"\n{'='*50}")
        print(f"Testing {tokenizer_type.upper()} tokenizer:")
        print(f"{'='*50}")
        
        try:
            tokenized_df, stats = tokenize_twitter_data(
                df_test, 
                tokenizer_type=tokenizer_type,
                vocab_size=500,  # Reduced from 1000
                max_length=50,
                model_path=f"../models/tokenizer_model_{tokenizer_type}"
            )
            
            if not tokenized_df.empty:
                print(f"\nSample tokenized results:")
                for i, row in tokenized_df.head(3).iterrows():
                    print(f"  Original: {row['sentence']}")
                    print(f"  Tokens: {row['tokens']}")
                    print(f"  Count: {row['token_count']}")
                    print()
            
            print(f"Statistics: {stats}")
            
        except Exception as e:
            print(f"❌ Error testing {tokenizer_type}: {e}")
            continue 