#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bitcoin Price Prediction Pipeline
Dự đoán giá Bitcoin thông qua phân tích cảm xúc Twitter

Author: Your Name
Date: 2025-09-19
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from configparser import ConfigParser
import warnings
warnings.filterwarnings('ignore')

# Import các module local (sẽ implement sau)
# from sentiment_analysis import analyze_sentiment
# from feature_engineering import create_features
# from model_training import train_models
# from evaluation import evaluate_models

def load_config():
    """Load configuration từ config.ini"""
    print("🔧 Loading configuration...")
    config = ConfigParser()
    try:
        with open('config.ini', 'r', encoding='utf-8') as f:
            config.read_file(f)
        print("   ✅ Config loaded successfully")
        return config
    except Exception as e:
        print(f"   ❌ Error loading config: {e}")
        sys.exit(1)

def load_data(config):
    """Load dữ liệu Bitcoin và Twitter"""
    print("\n📊 Loading datasets...")
    
    # Load Twitter data
    twitter_file = config.get('File', 'filename', fallback='bitcoin_tweets_1month_20250919_161545.csv')
    print(f"   📱 Loading Twitter data: {twitter_file}")
    tweets_path = os.path.join('data', 'tweets', twitter_file)
    if os.path.exists(tweets_path):
        twitter_df = pd.read_csv(tweets_path, encoding='utf-8')
        print(f"   ✅ Twitter data loaded: {len(twitter_df)} tweets")
        print(f"      - Columns: {list(twitter_df.columns)}")
        print(f"      - Date range: {twitter_df['date'].min()} to {twitter_df['date'].max()}" if 'date' in twitter_df.columns else "")
    else:
        print(f"   ❌ Twitter file not found: {tweets_path}")
        twitter_df = pd.DataFrame()
    
    # Load Bitcoin data (find latest file)
    bitcoin_files = [f for f in os.listdir('.') if f.startswith('bitcoin_price_data_') and f.endswith('.csv')]
    if bitcoin_files:
        bitcoin_file = sorted(bitcoin_files)[-1]  # Latest file
        print(f"   ₿ Loading Bitcoin data: {bitcoin_file}")
        
        bitcoin_df = pd.read_csv(bitcoin_file, encoding='utf-8')
        print(f"   ✅ Bitcoin data loaded: {len(bitcoin_df)} records")
        print(f"      - Columns: {list(bitcoin_df.columns)}")
        print(f"      - Date range: {bitcoin_df['date'].min()} to {bitcoin_df['date'].max()}" if 'date' in bitcoin_df.columns else "")
    else:
        print("   ❌ No Bitcoin data files found")
        bitcoin_df = pd.DataFrame()
    
    return twitter_df, bitcoin_df

def data_preprocessing(twitter_df, bitcoin_df):
    """Tiền xử lý dữ liệu"""
    print("\n🧹 Data Preprocessing...")
    
    # Twitter data preprocessing using advanced preprocessing pipeline
    print("   📱 Processing Twitter data...")
    if not twitter_df.empty:
        print(f"      - Original tweets: {len(twitter_df)}")
        
        # Import and use the preprocessing pipeline
        try:
            from preprocessing import preprocess_twitter_data
            
            # Configure preprocessing pipeline
            preprocessing_config = {
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
                'sentence_segmentation': {
                    'method': 'hybrid'
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
                    'save_dir': 'data/embeddings',
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
                    'output_dir': 'data/preprocessed'
                }
            }
            
            # Run preprocessing pipeline
            preprocessing_results = preprocess_twitter_data(
                twitter_df, 
                text_column='text',
                config=preprocessing_config
            )
            
            # Extract processed data
            if 'tokenized_data' in preprocessing_results.get('data', {}):
                processed_twitter = preprocessing_results['data']['tokenized_data']
                print(f"      ✅ Advanced preprocessing completed: {len(processed_twitter)} processed sentences")
                print(f"      - Generated embeddings: {list(preprocessing_results.get('data', {}).get('embeddings', {}).keys())}")
                print(f"      - Vocabulary size: {preprocessing_results.get('statistics', {}).get('tokenization', {}).get('vocabulary_size', 'N/A')}")
                
                # Store preprocessing results for later use
                processed_twitter.preprocessing_results = preprocessing_results
            else:
                print("      ⚠️ Preprocessing pipeline failed, using original data")
                processed_twitter = twitter_df.copy()
                
        except ImportError as e:
            print(f"      ⚠️ Could not import preprocessing pipeline: {e}")
            print("      - Using basic preprocessing fallback")
            processed_twitter = twitter_df.copy()
        except Exception as e:
            print(f"      ⚠️ Preprocessing pipeline error: {e}")
            print("      - Using original data")
            processed_twitter = twitter_df.copy()
    else:
        processed_twitter = pd.DataFrame()
    
    # Bitcoin data preprocessing
    print("   ₿ Processing Bitcoin data...")
    if not bitcoin_df.empty:
        print(f"      - Original records: {len(bitcoin_df)}")
        
        # Basic Bitcoin data preprocessing
        processed_bitcoin = bitcoin_df.copy()
        
        # Parse datetime columns
        if 'date' in processed_bitcoin.columns:
            processed_bitcoin['date'] = pd.to_datetime(processed_bitcoin['date'])
            print("      ✅ Parsed datetime columns")
        
        # Handle missing values
        initial_nulls = processed_bitcoin.isnull().sum().sum()
        processed_bitcoin = processed_bitcoin.dropna()
        final_nulls = processed_bitcoin.isnull().sum().sum()
        if initial_nulls > 0:
            print(f"      ✅ Handled missing values: removed {initial_nulls - final_nulls} null entries")
        
        # Calculate price changes if price column exists
        price_columns = ['price', 'close', 'Close', 'Price']
        price_col = None
        for col in price_columns:
            if col in processed_bitcoin.columns:
                price_col = col
                break
        
        if price_col:
            processed_bitcoin['price_change'] = processed_bitcoin[price_col].pct_change()
            processed_bitcoin['price_change_abs'] = processed_bitcoin[price_col].diff()
            processed_bitcoin['price_direction'] = (processed_bitcoin['price_change'] > 0).astype(int)
            print(f"      ✅ Calculated price changes and direction from '{price_col}' column")
        
        print(f"      ✅ Bitcoin data processed: {len(processed_bitcoin)} records")
    else:
        processed_bitcoin = pd.DataFrame()
    
    return processed_twitter, processed_bitcoin

def sentiment_analysis(twitter_df):
    """Phân tích cảm xúc Twitter data"""
    print("\n🎭 Sentiment Analysis...")
    
    if twitter_df.empty:
        print("   ❌ No Twitter data for sentiment analysis")
        return pd.DataFrame()
    
    print(f"   📊 Analyzing sentiment for {len(twitter_df)} tweets...")
    
    # TODO: Implement sentiment analysis
    print("   🔍 [TODO] TextBlob sentiment analysis")
    print("      - [TODO] Calculate polarity (-1 to 1)")
    print("      - [TODO] Calculate subjectivity (0 to 1)")
    
    print("   🔍 [TODO] VADER sentiment analysis")
    print("      - [TODO] Calculate compound score")
    print("      - [TODO] Calculate pos/neg/neu scores")
    
    print("   📈 [TODO] Aggregate sentiment by time periods")
    print("      - [TODO] Hourly sentiment averages")
    print("      - [TODO] Daily sentiment trends")
    print("      - [TODO] Sentiment volatility metrics")
    
    # Placeholder result
    sentiment_df = twitter_df.copy()
    sentiment_df['sentiment_polarity'] = np.random.uniform(-1, 1, len(sentiment_df))  # Mock data
    sentiment_df['sentiment_subjectivity'] = np.random.uniform(0, 1, len(sentiment_df))  # Mock data
    
    print(f"   ✅ Sentiment analysis completed")
    print(f"      - Average polarity: {sentiment_df['sentiment_polarity'].mean():.3f}")
    print(f"      - Average subjectivity: {sentiment_df['sentiment_subjectivity'].mean():.3f}")
    
    return sentiment_df

def data_synchronization(twitter_df, bitcoin_df):
    """Đồng bộ dữ liệu Twitter và Bitcoin theo thời gian"""
    print("\n🔄 Data Synchronization...")
    
    if twitter_df.empty or bitcoin_df.empty:
        print("   ❌ Missing data for synchronization")
        return pd.DataFrame()
    
    print("   ⏰ Synchronizing Twitter and Bitcoin data...")
    
    # TODO: Implement data synchronization
    print("   📅 [TODO] Parse and standardize timestamps")
    print("   🕐 [TODO] Aggregate Twitter sentiment by hour")
    print("   📊 [TODO] Merge with Bitcoin hourly data")
    print("   🔗 [TODO] Handle missing time periods")
    print("   📈 [TODO] Create lagged features")
    
    # Placeholder result
    print("   ✅ Data synchronization completed")
    print(f"      - [TODO] Synchronized records: XXX")
    print(f"      - [TODO] Time range: YYYY-MM-DD to YYYY-MM-DD")
    
    return pd.DataFrame()  # Placeholder

def feature_engineering(merged_df):
    """Tạo features cho machine learning"""
    print("\n⚙️ Feature Engineering...")
    
    if merged_df.empty:
        print("   ❌ No merged data for feature engineering")
        return pd.DataFrame(), []
    
    print("   🔧 Creating features...")
    
    # TODO: Implement feature engineering
    print("   📈 [TODO] Technical indicators")
    print("      - [TODO] Moving averages (MA5, MA10, MA20)")
    print("      - [TODO] RSI (Relative Strength Index)")
    print("      - [TODO] MACD indicators")
    print("      - [TODO] Bollinger Bands")
    
    print("   🎭 [TODO] Sentiment features")
    print("      - [TODO] Sentiment moving averages")
    print("      - [TODO] Sentiment volatility")
    print("      - [TODO] Sentiment momentum")
    
    print("   💰 [TODO] Price features")
    print("      - [TODO] Price returns")
    print("      - [TODO] Log returns")
    print("      - [TODO] Volatility measures")
    
    print("   🕐 [TODO] Temporal features")
    print("      - [TODO] Hour of day")
    print("      - [TODO] Day of week")
    print("      - [TODO] Time-based lags")
    
    feature_names = ['feature_1', 'feature_2', 'feature_3']  # Placeholder
    print(f"   ✅ Feature engineering completed")
    print(f"      - [TODO] Total features created: {len(feature_names)}")
    
    return pd.DataFrame(), feature_names  # Placeholder

def model_training(X, y, feature_names):
    """Training các mô hình machine learning"""
    print("\n🤖 Model Training...")
    
    if len(X) == 0 or len(y) == 0:
        print("   ❌ No data for model training")
        return {}
    
    print(f"   📊 Training data: {len(X)} samples, {len(feature_names)} features")
    
    # TODO: Implement model training
    print("   🔄 [TODO] Train/Test split (80/20)")
    print("   📊 [TODO] Feature scaling/normalization")
    
    print("   🧠 [TODO] Training models:")
    print("      - [TODO] Random Forest")
    print("      - [TODO] XGBoost")
    print("      - [TODO] LSTM Neural Network")
    print("      - [TODO] Support Vector Machine")
    print("      - [TODO] Linear Regression (baseline)")
    
    print("   ⚙️ [TODO] Hyperparameter tuning")
    print("   📈 [TODO] Cross-validation")
    
    models = {}  # Placeholder
    print("   ✅ Model training completed")
    
    return models

def model_evaluation(models, X_test, y_test):
    """Đánh giá hiệu suất các mô hình"""
    print("\n📊 Model Evaluation...")
    
    if not models:
        print("   ❌ No trained models to evaluate")
        return {}
    
    # TODO: Implement model evaluation
    print("   📈 [TODO] Evaluation metrics:")
    print("      - [TODO] Accuracy")
    print("      - [TODO] Precision/Recall/F1-Score")
    print("      - [TODO] ROC AUC")
    print("      - [TODO] Mean Squared Error")
    print("      - [TODO] Mean Absolute Error")
    
    print("   📊 [TODO] Model comparison")
    print("   🎯 [TODO] Feature importance analysis")
    print("   📈 [TODO] Learning curves")
    print("   🔍 [TODO] Error analysis")
    
    results = {}  # Placeholder
    print("   ✅ Model evaluation completed")
    
    return results

def generate_predictions(best_model, X_future):
    """Tạo dự đoán cho tương lai"""
    print("\n🔮 Generating Predictions...")
    
    # TODO: Implement prediction generation
    print("   🎯 [TODO] Generate future predictions")
    print("   📊 [TODO] Confidence intervals")
    print("   📈 [TODO] Trend analysis")
    print("   💾 [TODO] Save predictions to file")
    
    print("   ✅ Predictions generated")

def create_visualizations(twitter_df, bitcoin_df, results):
    """Tạo các biểu đồ và visualizations"""
    print("\n📈 Creating Visualizations...")
    
    # TODO: Implement visualizations
    print("   📊 [TODO] Data exploration plots")
    print("      - [TODO] Bitcoin price trends")
    print("      - [TODO] Sentiment distribution")
    print("      - [TODO] Correlation heatmap")
    
    print("   🎭 [TODO] Sentiment analysis plots")
    print("      - [TODO] Sentiment vs Price correlation")
    print("      - [TODO] Time series sentiment trends")
    
    print("   🤖 [TODO] Model performance plots")
    print("      - [TODO] Model comparison charts")
    print("      - [TODO] Feature importance plots")
    print("      - [TODO] Prediction vs Actual")
    
    print("   💾 [TODO] Save plots to charts/ directory")
    print("   ✅ Visualizations created")

def generate_report(results):
    """Tạo báo cáo tổng kết"""
    print("\n📝 Generating Final Report...")
    
    # TODO: Implement report generation
    print("   📊 [TODO] Summary statistics")
    print("   🎯 [TODO] Model performance summary")
    print("   📈 [TODO] Key insights and findings")
    print("   🔍 [TODO] Recommendations")
    print("   💾 [TODO] Save report to file")
    
    print("   ✅ Report generated")

def main():
    """Main pipeline function"""
    print("🚀 BITCOIN PRICE PREDICTION PIPELINE")
    print("=" * 60)
    print("📅 Started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("🎯 Goal: Predict Bitcoin price using Twitter sentiment")
    print("=" * 60)
    
    try:
        # 1. Load configuration
        config = load_config()
        
        # 2. Load data
        twitter_df, bitcoin_df = load_data(config)
        
        # 3. Data preprocessing
        twitter_df, bitcoin_df = data_preprocessing(twitter_df, bitcoin_df)
        
        # 4. Sentiment analysis
        twitter_with_sentiment = sentiment_analysis(twitter_df)
        
        # 5. Data synchronization
        merged_df = data_synchronization(twitter_with_sentiment, bitcoin_df)
        
        # 6. Feature engineering
        X, feature_names = feature_engineering(merged_df)
        
        # 7. Prepare target variable (placeholder)
        y = []  # TODO: Extract from merged_df
        
        # 8. Model training
        models = model_training(X, y, feature_names)
        
        # 9. Model evaluation
        results = model_evaluation(models, [], [])  # TODO: Pass test data
        
        # 10. Generate predictions
        generate_predictions(None, [])  # TODO: Pass best model and future data
        
        # 11. Create visualizations
        create_visualizations(twitter_df, bitcoin_df, results)
        
        # 12. Generate report
        generate_report(results)
        
        print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("📅 Finished at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("📊 Check output files and charts/ directory for results")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n⏹️ Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()