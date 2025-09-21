import pandas as pd
import numpy as np
from configparser import ConfigParser
import requests
import time
from datetime import datetime, timedelta
import json

config = ConfigParser()
try:
    with open('config.ini', 'r', encoding='utf-8') as f:
        config.read_file(f)
except FileNotFoundError:
    config.read('config.ini')  # fallback

def get_bitcoin_price_coingecko(start_date, end_date):
    """
    Fetch Bitcoin price data from CoinGecko API (free tier)
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    
    Returns:
        pandas.DataFrame: Bitcoin price data
    """
    try:
        # Convert dates to timestamps
        start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
        
        # CoinGecko API endpoint for historical data
        url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
        params = {
            'vs_currency': 'usd',
            'from': start_timestamp,
            'to': end_timestamp
        }
        
        print(f"Fetching Bitcoin price data from CoinGecko...")
        print(f"Date range: {start_date} to {end_date}")
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract price data
        prices = data.get('prices', [])
        volumes = data.get('total_volumes', [])
        market_caps = data.get('market_caps', [])
        
        if not prices:
            print("No price data received from CoinGecko")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df_data = []
        for i, price_point in enumerate(prices):
            timestamp_ms, price = price_point
            timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
            
            row = {
                'timestamp': timestamp,
                'date': timestamp.date(),
                'price_usd': price,
                'volume_usd': volumes[i][1] if i < len(volumes) else None,
                'market_cap_usd': market_caps[i][1] if i < len(market_caps) else None
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        print(f"Successfully fetched {len(df)} price records from CoinGecko")
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"CoinGecko API error: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error processing CoinGecko data: {e}")
        return pd.DataFrame()

def get_bitcoin_price_binance(start_date, end_date):
    """
    Fetch current Bitcoin price from Binance API as fallback
    
    Returns:
        pandas.DataFrame: Current Bitcoin price data
    """
    try:
        print("Fetching current Bitcoin price from Binance...")
        
        # Binance API for current price
        url = "https://api.binance.com/api/v3/ticker/24hr"
        params = {'symbol': 'BTCUSDT'}
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        current_time = datetime.now()
        df_data = [{
            'timestamp': current_time,
            'date': current_time.date(),
            'price_usd': float(data['lastPrice']),
            'volume_usd': float(data['volume']) * float(data['lastPrice']),
            'market_cap_usd': None,
            'price_change_24h': float(data['priceChange']),
            'price_change_percent_24h': float(data['priceChangePercent']),
            'high_24h': float(data['highPrice']),
            'low_24h': float(data['lowPrice'])
        }]
        
        df = pd.DataFrame(df_data)
        print(f"Successfully fetched current Bitcoin price from Binance: ${data['lastPrice']}")
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Binance API error: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error processing Binance data: {e}")
        return pd.DataFrame()

def fetch_bitcoin_data():
    """
    Main function to fetch Bitcoin price data according to config settings
    """
    try:
        # Read crypto configuration
        crypto_section = 'Crypto'
        start_date = config.get(crypto_section, 'start_date', fallback='2025-09-07')
        end_date = config.get(crypto_section, 'end_date', fallback='2025-09-11')
        
        print("Starting Bitcoin price data collection...")
        print("=" * 50)
        print(f"Configuration:")
        print(f"   Start Date: {start_date}")
        print(f"   End Date: {end_date}")
        print("=" * 50)
        
        # Try CoinGecko first (better for historical data)
        df = get_bitcoin_price_coingecko(start_date, end_date)
        
        # If CoinGecko fails, try Binance for current data
        if df.empty:
            print("Falling back to Binance API for current data...")
            df = get_bitcoin_price_binance(start_date, end_date)
        
        if df.empty:
            print("Failed to fetch data from all sources")
            return None
        
        # Add additional calculated fields
        df['source'] = 'coingecko' if 'price_change_24h' not in df.columns else 'binance'
        df['fetch_timestamp'] = datetime.now()
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Generate filename
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Ensure data/bitcoin-price directory exists
        import os
        os.makedirs("data/bitcoin-price", exist_ok=True)
        filename = f"data/bitcoin-price/bitcoin_price_data_{timestamp_str}.csv"
        
        # Export to CSV
        df.to_csv(filename, index=False)
        print(f"Data exported to: {filename}")
        
        # Display summary
        print("\nData Summary:")
        print("=" * 30)
        print(f"Records: {len(df)}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        if 'price_usd' in df.columns:
            print(f"Price range: ${df['price_usd'].min():.2f} - ${df['price_usd'].max():.2f}")
            print(f"Average price: ${df['price_usd'].mean():.2f}")
        
        print(f"\nSample data:")
        print(df.head().to_string())
        
        return df
        
    except Exception as e:
        print(f"Error in fetch_bitcoin_data: {e}")
        return None

if __name__ == "__main__":
    result = fetch_bitcoin_data()
    if result is not None:
        print("\nBitcoin price data collection completed successfully!")
    else:
        print("\nBitcoin price data collection failed!") 