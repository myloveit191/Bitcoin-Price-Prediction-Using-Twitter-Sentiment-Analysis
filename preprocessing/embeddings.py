#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embeddings Generation Module for Bitcoin Price Prediction
Tạo embeddings từ tokenized text: Static Embeddings (Word2Vec, GloVe) và Contextual Embeddings

Author: Bitcoin Price Prediction Team
Date: 2025-09-20
"""

import os
import pickle
import pandas as pd
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import warnings
warnings.filterwarnings('ignore')

# Import tokenization module
from .tokenization import tokenize_twitter_data

class TwitterEmbeddingsGenerator:
    """Class để tạo embeddings cho Twitter data với Static và Contextual Embeddings"""
    
    def __init__(self):
        """Khởi tạo embeddings generator"""
        self.word2vec_model = None
        self.glove_model = None
        self.sentence_bert_model = None
        self.bert_model = None
        self.bert_tokenizer = None
        
        # Default parameters for Word2Vec
        self.word2vec_params = {
            'vector_size': 300,
            'window': 5,
            'min_count': 2,
            'workers': 4,
            'epochs': 10,
            'sg': 0  # CBOW
        }
        
        # Default parameters for GloVe
        self.glove_params = {
            'vector_size': 300,
            'window': 5,
            'min_count': 2,
            'workers': 4,
            'epochs': 10
        }
        
        # Default parameters for Contextual Embeddings
        self.contextual_params = {
            'sentence_bert_model': 'all-MiniLM-L6-v2',
            'bert_model': 'bert-base-uncased',
            'max_length': 512,
            'batch_size': 32
        }
    
    def prepare_texts_for_embeddings(self, df, tokens_column='tokens'):
        """
        Prepare texts for different embedding methods
        
        Args:
            df: DataFrame with tokenized data
            tokens_column: Column name containing tokens
            
        Returns:
            Dictionary with prepared texts for different methods
        """
        if df.empty or tokens_column not in df.columns:
            print(f"   ❌ Column '{tokens_column}' not found or DataFrame is empty")
            return {}
        
        prepared_texts = {}
        
        # For Word2Vec and GloVe (list of token lists)
        prepared_texts['word_embeddings'] = df[tokens_column].apply(
            lambda tokens: tokens if isinstance(tokens, list) else str(tokens).split()
        ).tolist()
        
        # For Contextual Embeddings (original sentences if available, otherwise joined tokens)
        if 'sentence' in df.columns:
            prepared_texts['contextual'] = df['sentence'].tolist()
        else:
            prepared_texts['contextual'] = df[tokens_column].apply(
                lambda tokens: ' '.join(tokens) if isinstance(tokens, list) else str(tokens)
            ).tolist()
        
        return prepared_texts
    
    def generate_word2vec_embeddings(self, token_lists, **params):
        """
        Generate Word2Vec embeddings
        
        Args:
            token_lists: List of token lists
            **params: Word2Vec parameters
            
        Returns:
            Tuple of (embeddings_matrix, model)
        """
        print("   🧠 Generating Word2Vec embeddings...")
        
        if not token_lists or not any(token_lists):
            print("   ❌ No token lists provided for Word2Vec")
            return np.array([]), None
        
        # Update parameters
        w2v_params = self.word2vec_params.copy()
        w2v_params.update(params)
        
        try:
            # Train Word2Vec model
            self.word2vec_model = Word2Vec(sentences=token_lists, **w2v_params)
            
            # Generate sentence embeddings by averaging word vectors
            embeddings = []
            vector_size = w2v_params['vector_size']
            
            for tokens in token_lists:
                if not tokens:
                    # Empty token list - use zero vector
                    embeddings.append(np.zeros(vector_size))
                    continue
                
                # Get vectors for tokens that exist in vocabulary
                vectors = []
                for token in tokens:
                    if token in self.word2vec_model.wv:
                        vectors.append(self.word2vec_model.wv[token])
                
                if vectors:
                    # Average the word vectors
                    sentence_vector = np.mean(vectors, axis=0)
                else:
                    # No words found in vocabulary - use zero vector
                    sentence_vector = np.zeros(vector_size)
                
                embeddings.append(sentence_vector)
            
            embeddings_matrix = np.array(embeddings)
            
            print(f"   ✅ Word2Vec embeddings generated")
            print(f"      - Shape: {embeddings_matrix.shape}")
            print(f"      - Vocabulary size: {len(self.word2vec_model.wv.key_to_index)}")
            print(f"      - Vector size: {vector_size}")
            
            return embeddings_matrix, self.word2vec_model
            
        except Exception as e:
            print(f"   ❌ Word2Vec generation failed: {e}")
            return np.array([]), None
    
    def load_glove_embeddings(self, glove_path, **params):
        """
        Load pre-trained GloVe embeddings
        
        Args:
            glove_path: Path to GloVe embeddings file
            **params: GloVe parameters
            
        Returns:
            Tuple of (embeddings_matrix, model)
        """
        print(f"   📚 Loading GloVe embeddings from {glove_path}...")
        
        if not os.path.exists(glove_path):
            print(f"   ❌ GloVe file not found: {glove_path}")
            return np.array([]), None
        
        try:
            # Load GloVe embeddings
            self.glove_model = KeyedVectors.load_word2vec_format(glove_path, binary=False, no_header=True)
            
            print(f"   ✅ GloVe embeddings loaded")
            print(f"      - Vocabulary size: {len(self.glove_model.key_to_index)}")
            print(f"      - Vector size: {self.glove_model.vector_size}")
            
            return self.glove_model
            
        except Exception as e:
            print(f"   ❌ GloVe loading failed: {e}")
            return None
    
    def generate_glove_embeddings(self, token_lists, glove_path=None, **params):
        """
        Generate GloVe embeddings for token lists
        
        Args:
            token_lists: List of token lists
            glove_path: Path to pre-trained GloVe embeddings
            **params: GloVe parameters
            
        Returns:
            Tuple of (embeddings_matrix, model)
        """
        print("   📚 Generating GloVe embeddings...")
        
        if not token_lists or not any(token_lists):
            print("   ❌ No token lists provided for GloVe")
            return np.array([]), None
        
        # Load GloVe model if path provided
        if glove_path:
            glove_model = self.load_glove_embeddings(glove_path, **params)
            if glove_model is None:
                return np.array([]), None
        else:
            print("   ⚠️ No GloVe path provided, skipping GloVe embeddings")
            return np.array([]), None
        
        try:
            # Generate sentence embeddings by averaging word vectors
            embeddings = []
            vector_size = glove_model.vector_size
            
            for tokens in token_lists:
                if not tokens:
                    # Empty token list - use zero vector
                    embeddings.append(np.zeros(vector_size))
                    continue
                
                # Get vectors for tokens that exist in vocabulary
                vectors = []
                for token in tokens:
                    if token in glove_model:
                        vectors.append(glove_model[token])
                
                if vectors:
                    # Average the word vectors
                    sentence_vector = np.mean(vectors, axis=0)
                else:
                    # No words found in vocabulary - use zero vector
                    sentence_vector = np.zeros(vector_size)
                
                embeddings.append(sentence_vector)
            
            embeddings_matrix = np.array(embeddings)
            
            print(f"   ✅ GloVe embeddings generated")
            print(f"      - Shape: {embeddings_matrix.shape}")
            print(f"      - Vector size: {vector_size}")
            
            return embeddings_matrix, glove_model
            
        except Exception as e:
            print(f"   ❌ GloVe generation failed: {e}")
            return np.array([]), None
    
    def generate_sentence_bert_embeddings(self, sentences, model_name='all-MiniLM-L6-v2'):
        """
        Generate Sentence-BERT embeddings (Contextual)
        
        Args:
            sentences: List of sentence strings
            model_name: Pre-trained SBERT model name
            
        Returns:
            Tuple of (embeddings_matrix, model)
        """
        print(f"    Generating Sentence-BERT embeddings using {model_name}...")
        
        if not sentences:
            print("   ❌ No sentences provided for Sentence-BERT")
            return np.array([]), None
        
        try:
            # Load pre-trained Sentence-BERT model
            self.sentence_bert_model = SentenceTransformer(model_name)
            
            # Generate embeddings
            embeddings = self.sentence_bert_model.encode(sentences, 
                                                       show_progress_bar=True,
                                                       batch_size=32)
            
            print(f"   ✅ Sentence-BERT embeddings generated")
            print(f"      - Shape: {embeddings.shape}")
            print(f"      - Model: {model_name}")
            print(f"      - Vector size: {embeddings.shape[1]}")
            
            return embeddings, self.sentence_bert_model
            
        except Exception as e:
            print(f"   ❌ Sentence-BERT generation failed: {e}")
            print("   💡 Installing sentence-transformers might be needed: pip install sentence-transformers")
            return np.array([]), None
    
    def generate_bert_embeddings(self, sentences, model_name='bert-base-uncased', max_length=512):
        """
        Generate BERT embeddings (Contextual)
        
        Args:
            sentences: List of sentence strings
            model_name: Pre-trained BERT model name
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (embeddings_matrix, model)
        """
        print(f"    Generating BERT embeddings using {model_name}...")
        
        if not sentences:
            print("   ❌ No sentences provided for BERT")
            return np.array([]), None
        
        try:
            # Load pre-trained BERT model and tokenizer
            self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert_model = AutoModel.from_pretrained(model_name)
            
            # Set device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.bert_model.to(device)
            self.bert_model.eval()
            
            embeddings = []
            batch_size = 32
            
            for i in range(0, len(sentences), batch_size):
                batch_sentences = sentences[i:i + batch_size]
                
                # Tokenize batch
                inputs = self.bert_tokenizer(
                    batch_sentences,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                )
                
                # Move to device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Get embeddings
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    # Use [CLS] token embeddings (first token)
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    embeddings.extend(batch_embeddings)
            
            embeddings_matrix = np.array(embeddings)
            
            print(f"   ✅ BERT embeddings generated")
            print(f"      - Shape: {embeddings_matrix.shape}")
            print(f"      - Model: {model_name}")
            print(f"      - Vector size: {embeddings_matrix.shape[1]}")
            
            return embeddings_matrix, self.bert_model
            
        except Exception as e:
            print(f"   ❌ BERT generation failed: {e}")
            print("   💡 Installing transformers might be needed: pip install transformers")
            return np.array([]), None
    
    def generate_all_embeddings(self, df, 
                              tokens_column='tokens',
                              generate_word2vec=True,
                              generate_glove=True,
                              generate_sbert=True,
                              generate_bert=True,
                              glove_path=None,
                              **embedding_params):
        """
        Generate all types of embeddings
        
        Args:
            df: DataFrame with tokenized data
            tokens_column: Column name containing tokens
            generate_word2vec: Whether to generate Word2Vec embeddings
            generate_glove: Whether to generate GloVe embeddings
            generate_sbert: Whether to generate Sentence-BERT embeddings
            generate_bert: Whether to generate BERT embeddings
            glove_path: Path to pre-trained GloVe embeddings
            **embedding_params: Parameters for specific embedding methods
            
        Returns:
            Dictionary containing all embeddings and models
        """
        print(f"🎯 Generating embeddings for {len(df)} texts...")
        
        # Prepare texts
        prepared_texts = self.prepare_texts_for_embeddings(df, tokens_column)
        
        if not prepared_texts:
            return {}
        
        results = {}
        
        # Generate Word2Vec embeddings
        if generate_word2vec:
            w2v_params = embedding_params.get('word2vec_params', {})
            w2v_embeddings, w2v_model = self.generate_word2vec_embeddings(
                prepared_texts['word_embeddings'], **w2v_params
            )
            results['word2vec'] = {
                'embeddings': w2v_embeddings,
                'model': w2v_model
            }
        
        # Generate GloVe embeddings
        if generate_glove and glove_path:
            glove_params = embedding_params.get('glove_params', {})
            glove_embeddings, glove_model = self.generate_glove_embeddings(
                prepared_texts['word_embeddings'], glove_path=glove_path, **glove_params
            )
            results['glove'] = {
                'embeddings': glove_embeddings,
                'model': glove_model
            }
        
        # Generate Sentence-BERT embeddings
        if generate_sbert:
            sbert_params = embedding_params.get('sbert_params', {})
            sbert_model_name = sbert_params.get('model_name', 'all-MiniLM-L6-v2')
            sbert_embeddings, sbert_model = self.generate_sentence_bert_embeddings(
                prepared_texts['contextual'], model_name=sbert_model_name
            )
            results['sentence_bert'] = {
                'embeddings': sbert_embeddings,
                'model': sbert_model
            }
        
        # Generate BERT embeddings
        if generate_bert:
            bert_params = embedding_params.get('bert_params', {})
            bert_model_name = bert_params.get('model_name', 'bert-base-uncased')
            bert_max_length = bert_params.get('max_length', 512)
            bert_embeddings, bert_model = self.generate_bert_embeddings(
                prepared_texts['contextual'], model_name=bert_model_name, max_length=bert_max_length
            )
            results['bert'] = {
                'embeddings': bert_embeddings,
                'model': bert_model
            }
        
        # Calculate statistics
        stats = self._calculate_embedding_stats(results)
        results['statistics'] = stats
        
        print(f"✅ All embeddings generated successfully")
        return results
    
    def _calculate_embedding_stats(self, results):
        """Calculate embedding statistics"""
        stats = {}
        
        for embedding_type, data in results.items():
            if embedding_type == 'statistics':
                continue
                
            embeddings = data['embeddings']
            if embeddings.size > 0:
                stats[embedding_type] = {
                    'shape': embeddings.shape,
                    'mean': np.mean(embeddings),
                    'std': np.std(embeddings),
                    'min': np.min(embeddings),
                    'max': np.max(embeddings),
                    'non_zero_ratio': np.count_nonzero(embeddings) / embeddings.size
                }
        
        return stats
    
    def save_embeddings(self, results, save_dir='embeddings'):
        """
        Save embeddings and models to disk
        
        Args:
            results: Results dictionary from generate_all_embeddings
            save_dir: Directory to save embeddings
        """
        print(f"💾 Saving embeddings to {save_dir}/...")
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        for embedding_type, data in results.items():
            if embedding_type == 'statistics':
                continue
            
            embeddings = data['embeddings']
            model = data['model']
            
            if embeddings.size > 0:
                # Save embeddings as numpy array
                embeddings_path = os.path.join(save_dir, f'{embedding_type}_embeddings.npy')
                np.save(embeddings_path, embeddings)
                print(f"   ✅ Saved {embedding_type} embeddings: {embeddings_path}")
                
                # Save model
                if model is not None:
                    model_path = os.path.join(save_dir, f'{embedding_type}_model')
                    
                    try:
                        if embedding_type in ['word2vec', 'glove']:
                            # Gensim models have their own save method
                            model_path = f"{model_path}.model"
                            model.save(model_path)
                        elif embedding_type in ['sentence_bert', 'bert']:
                            # Hugging Face models
                            model.save_pretrained(model_path)
                        else:
                            # Other models
                            model_path = f"{model_path}.pkl"
                            with open(model_path, 'wb') as f:
                                pickle.dump(model, f)
                        
                        print(f"   ✅ Saved {embedding_type} model: {model_path}")
                        
                    except Exception as e:
                        print(f"   ⚠️ Failed to save {embedding_type} model: {e}")
        
        # Save statistics
        stats_path = os.path.join(save_dir, 'embedding_statistics.pkl')
        with open(stats_path, 'wb') as f:
            pickle.dump(results.get('statistics', {}), f)
        print(f"   ✅ Saved embedding statistics: {stats_path}")

def generate_twitter_embeddings(df, 
                              text_column='sentence',
                              tokens_column='tokens',
                              embedding_types=['word2vec', 'sentence_bert'],
                              tokenizer_type='sentencepiece',
                              vocab_size=8000,
                              model_path=None,
                              glove_path=None,
                              save_embeddings=False,
                              save_dir='embeddings',
                              **embedding_options):
    """
    Main function to generate Twitter embeddings with tokenization integration
    
    Args:
        df: DataFrame with Twitter data
        text_column: Column name containing original text
        tokens_column: Column name for tokenized text
        embedding_types: List of embedding types to generate
        tokenizer_type: Type of tokenizer to use
        vocab_size: Vocabulary size for tokenizer
        model_path: Path to pretrained tokenizer model
        glove_path: Path to pre-trained GloVe embeddings
        save_embeddings: Whether to save embeddings to disk
        save_dir: Directory to save embeddings
        **embedding_options: Options for embedding generation
        
    Returns:
        Dictionary containing all embeddings and models
    """
    print("🎯 Starting Twitter embeddings generation with tokenization...")
    
    if df.empty:
        print("   ❌ Empty DataFrame provided")
        return {}
    
    # Check if data is already tokenized
    if tokens_column not in df.columns or df[tokens_column].isna().all():
        print("   🔤 Tokenizing data using advanced tokenization...")
        
        # Tokenize data using the tokenization module
        tokenized_df, token_stats = tokenize_twitter_data(
            df, 
            text_column=text_column,
            tokenizer_type=tokenizer_type,
            vocab_size=vocab_size,
            model_path=model_path
        )
        
        print(f"   ✅ Tokenization completed: {token_stats}")
    else:
        print("   ✅ Using pre-tokenized data")
        tokenized_df = df
    
    # Initialize embeddings generator
    generator = TwitterEmbeddingsGenerator()
    
    # Set generation flags based on embedding_types
    generation_flags = {
        'generate_word2vec': 'word2vec' in embedding_types,
        'generate_glove': 'glove' in embedding_types,
        'generate_sbert': 'sentence_bert' in embedding_types,
        'generate_bert': 'bert' in embedding_types
    }
    
    # Generate embeddings
    results = generator.generate_all_embeddings(
        tokenized_df, tokens_column, glove_path=glove_path, **generation_flags, **embedding_options
    )
    
    # Add tokenization statistics to results
    if 'tokenization_stats' not in results:
        results['tokenization_stats'] = token_stats if 'token_stats' in locals() else {}
    
    # Save embeddings if requested
    if save_embeddings and results:
        generator.save_embeddings(results, save_dir)
    
    print("✅ Twitter embeddings generation completed")
    return results

if __name__ == "__main__":
    # Test the embeddings functions
    print("🧪 Testing Twitter Embeddings Generator...")
    
    # Sample data
    test_data = {
        'sentence': [
            'bitcoin is going up today',
            'i think it will reach new heights',
            'btc price analysis',
            'just bought some eth and btc',
            'hodling for the long term',
            'crypto market is volatile',
            'defi is the future'
        ]
    }
    
    df_test = pd.DataFrame(test_data)
    print(f"Test data shape: {df_test.shape}")
    
    # Generate embeddings
    results = generate_twitter_embeddings(
        df_test, 
        embedding_types=['word2vec', 'sentence_bert'],
        tokenizer_type='sentencepiece',
        vocab_size=1000,
        save_embeddings=False
    )
    
    print(f"\nGenerated embeddings:")
    for emb_type, data in results.items():
        if emb_type not in ['statistics', 'tokenization_stats']:
            print(f"- {emb_type}: {data['embeddings'].shape}")
    
    if 'statistics' in results:
        print(f"\nEmbedding statistics:")
        for emb_type, stats in results['statistics'].items():
            print(f"- {emb_type}: {stats['shape']}") 