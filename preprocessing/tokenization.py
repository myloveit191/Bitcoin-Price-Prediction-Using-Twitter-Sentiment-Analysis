#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tokenization Module for Bitcoin Price Prediction
Single BPE tokenization using pretrained GPT-2 tokenizer
Focus: Merge pretrained vocabulary with crypto domain-specific tokens

Author: Bitcoin Price Prediction Team
Date: 2025-09-20
"""

import re
import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from transformers import GPT2TokenizerFast
import warnings
warnings.filterwarnings('ignore')

class TwitterTokenizer:
    """BPE tokenizer for cleaned Twitter sentences using GPT-2 with domain vocab extension"""
    
    def __init__(self, model_name: str = 'gpt2', add_domain_tokens: bool = True):
        """
        Khởi tạo GPT-2 BPE tokenizer và mở rộng với crypto vocabulary
        
        Args:
            model_name: pretrained tokenizer name (e.g., 'gpt2')
            add_domain_tokens: có thêm crypto domain tokens vào vocab không
        """
        self.model_name = model_name
        self.add_domain_tokens = add_domain_tokens

        # Crypto vocabulary to preserve (được thêm vào added tokens)
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

        # HF tokenizer instance
        self.tokenizer: Optional[GPT2TokenizerFast] = None

        # Load or initialize tokenizer
        self._load_or_initialize_tokenizer()

    def _load_or_initialize_tokenizer(self) -> None:
        """Load pretrained GPT-2 tokenizer, extend with domain tokens, and ensure PAD token."""
        try:
            self.tokenizer = GPT2TokenizerFast.from_pretrained(self.model_name)
            print(f"   📥 GPT-2 tokenizer loaded: {self.model_name}")
        except Exception as e:
            print(f"   ❌ Error loading GPT-2 tokenizer: {e}")
            print("   💥 Exiting program due to tokenizer initialization failure")
            sys.exit(1)

        # Ensure PAD token exists (GPT-2 lacks PAD by default)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        # Add domain tokens as added tokens (preserve as whole tokens when present)
        if self.add_domain_tokens:
            # Lowercase variants for Twitter text
            domain_tokens = sorted({t.lower() for t in self.crypto_vocabulary})
            # Avoid adding if they already exist in vocab
            tokens_to_add = [t for t in domain_tokens if t not in self.tokenizer.get_vocab()]
            if tokens_to_add:
                num_added = self.tokenizer.add_tokens(tokens_to_add, special_tokens=False)
                print(f"   ➕ Added {num_added} domain tokens to tokenizer")

    def train_tokenizer(self, texts: List[str]) -> None:
        """
        For GPT-2 BPE we do not train from scratch here; we rely on pretrained vocab
        and extend with domain tokens. This method is retained for API compatibility.
        """
        if self.tokenizer is None:
            print("   ❌ Tokenizer not available")
            print("   💥 Exiting program due to tokenizer unavailability")
            sys.exit(1)
        print("✅ Tokenizer ready (pretrained GPT-2 with domain tokens)")

    def tokenize_single_text(self, text: str,
                           max_length: int = 512,
                           add_special_tokens: bool = True,
                           return_tokens: bool = True,
                           return_ids: bool = False) -> Any:
        """
        Tokenize một text đơn lẻ sử dụng GPT-2 BPE
        
        Returns dict if both tokens and ids requested, else list
        """
        if pd.isna(text) or not isinstance(text, str) or len(text.strip()) == 0:
            return {'tokens': [], 'ids': []} if return_ids else []
        
        text = text.strip().lower()
        result: Dict[str, Any] = {}

        if not self.tokenizer:
            print("   ❌ Tokenizer not available")
            print("   💥 Exiting program due to tokenizer unavailability")
            sys.exit(1)

        try:
            encoding = self.tokenizer(
                text,
                add_special_tokens=add_special_tokens,
                truncation=True if max_length else False,
                max_length=max_length,
                return_attention_mask=False,
                return_token_type_ids=False,
                return_offsets_mapping=False
            )

            if return_tokens:
                tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'])
                # Post-processing: loại bỏ token 'Ġ' đơn lẻ
                tokens = [token for token in tokens if token != 'Ġ']
                result['tokens'] = tokens
            if return_ids:
                result['ids'] = encoding['input_ids']
        
        except Exception as e:
            print(f"   ❌ Error in tokenization: {e}")
            print("   💥 Exiting program due to tokenization failure")
            sys.exit(1)
        
        if return_tokens and return_ids:
            return result
        elif return_tokens:
            return result.get('tokens', [])
        elif return_ids:
            return result.get('ids', [])
        else:
            return result.get('tokens', [])
    
    def tokenize_dataframe(self, df: pd.DataFrame, text_column: str = 'sentence', **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Tokenize text data trong pandas DataFrame bằng GPT-2 BPE
        """
        if df.empty or text_column not in df.columns:
            print(f"   ❌ Column '{text_column}' not found or DataFrame is empty")
            print("   💥 Exiting program due to invalid input data")
            sys.exit(1)
        
        print(f"   🔤 Tokenizing {len(df)} texts using GPT-2 BPE tokenizer...")

        if not self.tokenizer:
            print("   ❌ Tokenizer not available")
            print("   💥 Exiting program due to tokenizer unavailability")
            sys.exit(1)

        tokenized_data: List[pd.Series] = []
        total_tokens = 0
        
        for idx, row in df.iterrows():
            text = row[text_column]
            token_result = self.tokenize_single_text(text, **kwargs)
            
            if isinstance(token_result, dict):
                tokens = token_result.get('tokens', [])
                token_ids = token_result.get('ids', [])
            else:
                tokens = token_result
                token_ids = []
            
            new_row = row.copy()
            new_row['tokens'] = tokens
            new_row['token_ids'] = token_ids
            new_row['token_count'] = len(tokens)
            new_row['tokens_string'] = ' '.join(tokens) if tokens else ''
            tokenized_data.append(new_row)
            
            total_tokens += len(tokens)
            
            if (idx + 1) % 1000 == 0:
                print(f"      - Processed {idx + 1}/{len(df)} texts")
        
        df_tokenized = pd.DataFrame(tokenized_data)
        
        original_length = len(df_tokenized)
        df_tokenized = df_tokenized[df_tokenized['token_count'] > 0]
        final_length = len(df_tokenized)
        
        stats = self._calculate_tokenization_stats(df_tokenized, total_tokens, 
                                                 original_length, final_length)
        
        print(f"   ✅ Tokenization completed")
        print(f"      - Tokenizer type: GPT-2 BPE")
        print(f"      - Original texts: {original_length}")
        print(f"      - Final texts: {final_length}")
        print(f"      - Total tokens: {total_tokens}")
        print(f"      - Avg tokens per text: {stats['avg_tokens_per_text']:.1f}")
        
        return df_tokenized, stats

    def _calculate_tokenization_stats(self, df: pd.DataFrame, total_tokens: int, original_length: int, final_length: int) -> Dict[str, Any]:
        """Calculate tokenization statistics"""
        stats: Dict[str, Any] = {
            'tokenizer_type': 'gpt2-bpe',
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
            all_tokens: List[str] = []
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
    
    def save_model(self, save_path: str) -> None:
        """Lưu tokenizer (Hugging Face format directory)."""
        if not save_path:
            print("   ❌ No save path provided")
            print("   💥 Exiting program due to invalid save path")
            sys.exit(1)
        
        os.makedirs(save_path, exist_ok=True)
        try:
            if self.tokenizer:
                self.tokenizer.save_pretrained(save_path)
                print(f"   💾 GPT-2 tokenizer saved at {save_path}")
            else:
                print("   ❌ No tokenizer to save")
                print("   💥 Exiting program due to missing tokenizer")
                sys.exit(1)
        except Exception as e:
            print(f"   ❌ Error saving tokenizer: {e}")
            print("   💥 Exiting program due to save failure")
            sys.exit(1)
    
    def load_model(self, model_name: str) -> None:
        """Load tokenizer từ Hugging Face Hub."""
        if not model_name:
            print("   ❌ No model name provided")
            print("   💥 Exiting program due to missing model name")
            sys.exit(1)
        self.model_name = model_name
        self._load_or_initialize_tokenizer()

    def create_vocabulary(self, df: pd.DataFrame, min_frequency: int = 2, max_vocab_size: int = 10000) -> Dict[str, int]:
        """
        Create vocabulary từ tokenized DataFrame (analytical; independent of HF vocab)
        """
        if df.empty or 'tokens' not in df.columns:
            print("   ❌ No tokenized data found")
            print("   💥 Exiting program due to invalid input data")
            sys.exit(1)
        
        print(f"   📚 Creating vocabulary from {len(df)} texts...")
        
        from collections import Counter
        all_tokens: List[str] = []
        for tokens in df['tokens']:
            if isinstance(tokens, list):
                all_tokens.extend(tokens)
        
        token_counter = Counter(all_tokens)
        
        filtered_tokens = [token for token, freq in token_counter.items() \
                          if freq >= min_frequency]
        
        sorted_tokens = sorted(filtered_tokens, 
                             key=lambda x: token_counter[x], 
                             reverse=True)[:max_vocab_size]
        
        vocab = {'<PAD>': 0, '<UNK>': 1, '<CLS>': 2, '<SEP>': 3}
        for i, token in enumerate(sorted_tokens):
            vocab[token] = i + 4
        
        print(f"   ✅ Vocabulary created")
        print(f"      - Total unique tokens: {len(token_counter)}")
        print(f"      - Filtered tokens (freq >= {min_frequency}): {len(filtered_tokens)}")
        print(f"      - Final vocabulary size: {len(vocab)}")
        
        return vocab


def tokenize_twitter_data(df: pd.DataFrame, text_column: str = 'sentence', 
                         model_name: str = 'gpt2', add_domain_tokens: bool = True, 
                         **tokenization_options) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Main function để tokenize Twitter data với single BPE (GPT-2)
    
    Args:
        df: Input DataFrame
        text_column: Name of text column
        model_name: Pretrained tokenizer name (e.g., 'gpt2')
        add_domain_tokens: Whether to add crypto domain tokens
        **tokenization_options: Additional options
    
    Returns:
        Tuple of (tokenized_dataframe, tokenization_statistics)
    """
    print(f"🔤 Starting BPE tokenization with GPT-2 ({model_name})...")
    
    if df.empty:
        print("   ❌ Empty DataFrame provided")
        print("   💥 Exiting program due to empty input data")
        sys.exit(1)
    
    tokenizer = TwitterTokenizer(
        model_name=model_name,
        add_domain_tokens=add_domain_tokens
    )
    
    default_options = {
        'max_length': 512,
        'add_special_tokens': True,
        'return_tokens': True,
        'return_ids': False
    }
    
    default_options.update(tokenization_options)
    
    tokenized_df, stats = tokenizer.tokenize_dataframe(df, text_column, **default_options)
    
    print(f"✅ BPE tokenization completed with GPT-2")
    return tokenized_df, stats

if __name__ == "__main__":
    # Test BPE tokenization
    print("🧪 Testing GPT-2 BPE Twitter Tokenizer...")
    
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
    
    try:
        tokenized_df, stats = tokenize_twitter_data(
            df_test,
            model_name='gpt2',
            add_domain_tokens=True,
            max_length=50
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
        print(f"❌ Error testing GPT-2 BPE: {e}")
        print("💥 Exiting program due to test failure")
        sys.exit(1) 