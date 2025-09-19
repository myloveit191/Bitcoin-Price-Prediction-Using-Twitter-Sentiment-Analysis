import asyncio
from twikit import Client, TooManyRequests
import time
from datetime import datetime
import pandas as pd
import re
import html
from configparser import ConfigParser
from random import randint
import os
import json

# Cấu hình
# Load config
config = ConfigParser()
config.read('config.ini')
username = config['X']['username']
password = config['X']['password']
email = config['X']['email']

# Read scraping configuration
scraping_section = 'Scraping'
min_tweets = config.getint(scraping_section, 'min_tweets', fallback=1000)
start_date = config.get(scraping_section, 'start_date', fallback='2025-09-01')
end_date = config.get(scraping_section, 'end_date', fallback='2025-09-10')
keywords = config.get(scraping_section, 'keywords', fallback='bitcoin')
lang = config.get(scraping_section, 'lang', fallback='en')
provided_query = config.get(scraping_section, 'query', fallback='').strip()
product = config.get(scraping_section, 'product', fallback='Top')
count_per_request = config.getint(scraping_section, 'count_per_request', fallback=20)
request_delay_seconds = config.getint(scraping_section, 'request_delay_seconds', fallback=10)
max_rate_limit_retries = config.getint(scraping_section, 'max_rate_limit_retries', fallback=5)

# Build query if not explicitly provided
if provided_query:
    QUERY = provided_query
else:
    QUERY = f"{keywords} lang:{lang} since:{start_date} until:{end_date}"

def print_rate_limit_info():
    """
    🔍 NGHIÊN CỨU TWITTER RATE LIMIT ERROR 88:
    
    Error 88 = "Rate limit exceeded" 
    
    NGUYÊN NHÂN:
    - Twitter giới hạn 180 requests mỗi 15 phút cho search API
    - Khi vượt quá giới hạn này, API trả về error code 88
    - Cần đợi đến khi quota reset (thường 15 phút)
    
    GIẢI PHÁP HIỆU QUẢ:
    1. Exponential Backoff: Tăng thời gian đợi theo cấp số nhân
    2. Intelligent Rate Management: Theo dõi và quản lý quota
    3. Graceful Error Handling: Xử lý lỗi một cách thông minh
    4. Retry với Sleep: Đợi và thử lại khi gặp rate limit
    """
    print("🔍 NGHIÊN CỨU TWITTER RATE LIMIT ERROR 88:")
    print("=" * 60)
    print("Error 88 = 'Rate limit exceeded'")
    print("\nNGUYÊN NHÂN:")
    print("- Twitter giới hạn 180 requests mỗi 15 phút cho search API")
    print("- Khi vượt quá giới hạn này, API trả về error code 88")
    print("- Cần đợi đến khi quota reset (thường 15 phút)")
    print("\nGIẢI PHÁP HIỆU QUẢ:")
    print("1. Exponential Backoff: Tăng thời gian đợi theo cấp số nhân")
    print("2. Intelligent Rate Management: Theo dõi và quản lý quota")
    print("3. Graceful Error Handling: Xử lý lỗi một cách thông minh")
    print("4. Retry với Sleep: Đợi và thử lại khi gặp rate limit")
    print("=" * 60)

def clean_tweet_text(text):
    """
    Làm sạch text toàn diện:
    - Loại bỏ URLs (http, https, www)
    - Loại bỏ HTML entities và symbols
    - Loại bỏ ký tự đặc biệt và emoji
    - Loại bỏ mentions và hashtags symbols
    """
    if not text:
        return ""
    
    # Decode HTML entities
    text = html.unescape(text)
    
    # Loại bỏ URLs (bao gồm cả shortened URLs)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r't\.co/\S+', '', text)
    
    # Loại bỏ mentions và hashtag symbols (giữ lại text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)  # Giữ lại từ, bỏ dấu #
    
    # Loại bỏ emoji và ký tự đặc biệt
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    # Loại bỏ số và ký tự đặc biệt khác
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def handle_rate_limit_error(attempt, max_attempts=10):
    """
    Xử lý rate limit error với exponential backoff
    
    Args:
        attempt: Số lần thử hiện tại
        max_attempts: Số lần thử tối đa
    
    Returns:
        wait_time: Thời gian đợi (giây)
    """
    if attempt >= max_attempts:
        raise Exception(f"Đã thử {max_attempts} lần nhưng vẫn gặp rate limit. Dừng script.")
    
    # Exponential backoff: 2^attempt * 60 seconds (tối thiểu 1 phút, tối đa 15 phút)
    base_wait = min(60 * (2 ** attempt), 900)  # Max 15 phút
    
    # Thêm random jitter để tránh thundering herd
    jitter = randint(10, 60)
    wait_time = base_wait + jitter
    
    print(f"🚨 RATE LIMIT HIT! Attempt {attempt + 1}/{max_attempts}")
    print(f"⏰ Waiting {wait_time // 60}m {wait_time % 60}s before retry...")
    print(f"💡 Sử dụng exponential backoff strategy")
    
    return wait_time

async def robust_search_tweets(client, query, max_tweets, product='Top', count_per_request=20, request_delay_seconds=10, max_rate_limit_retries=5):
    """
    Tìm kiếm tweets với xử lý rate limit mạnh mẽ
    """
    tweets = []
    attempts = 0
    pagination_token = None
    
    print(f"🔍 Bắt đầu tìm kiếm tweets với query: '{query}'")
    print(f"🎯 Mục tiêu: {max_tweets} tweets")
    
    while len(tweets) < max_tweets and attempts < max_rate_limit_retries:
        try:
            print(f"\n📡 Đang gửi request... (Đã có {len(tweets)} tweets)")
            
            # Thực hiện search
            if pagination_token:
                search_results = await client.search_tweet(
                    query, 
                    product=product,
                    count=count_per_request,
                    cursor=pagination_token
                )
            else:
                search_results = await client.search_tweet(
                    query, 
                    product=product,
                    count=count_per_request
                )
            
            # Xử lý kết quả
            batch_tweets = []
            for tweet in search_results:
                if len(tweets) + len(batch_tweets) >= max_tweets:
                    break
                    
                batch_tweets.append({
                    'id': tweet.id,
                    'text': tweet.text,
                    'user': tweet.user.screen_name,
                    'created_at': tweet.created_at,
                    'retweet_count': tweet.retweet_count,
                    'favorite_count': tweet.favorite_count,
                    'reply_count': getattr(tweet, 'reply_count', 0)
                })
            
            tweets.extend(batch_tweets)
            print(f"✅ Thu thập được {len(batch_tweets)} tweets trong batch này")
            print(f"📊 Tổng cộng: {len(tweets)}/{max_tweets} tweets")
            
            # Lấy pagination token cho lần tiếp theo
            if hasattr(search_results, 'next_cursor') and search_results.next_cursor:
                pagination_token = search_results.next_cursor
            else:
                print("📄 Không còn trang tiếp theo")
                break
            
            # Đặt delay nhỏ giữa các request để tránh rate limit
            await asyncio.sleep(request_delay_seconds) # configured delay
            
        except TooManyRequests as e:
            attempts += 1
            print(f"\n🚨 RATE LIMIT ERROR (Code 88) detected!")
            
            wait_time = handle_rate_limit_error(attempts - 1, max_rate_limit_retries)
            
            print(f"⏳ Sleeping for {wait_time} seconds...")
            await asyncio.sleep(wait_time)
            
            print("🔄 Retrying after rate limit wait...")
            continue
            
        except Exception as e:
            attempts += 1
            error_str = str(e)
            
            # Check if this is a rate limit error in disguise
            if "rate limit exceeded" in error_str.lower() or "code\":88" in error_str:
                print(f"\n🚨 RATE LIMIT ERROR (Code 88) detected in exception!")
                wait_time = handle_rate_limit_error(attempts - 1, max_rate_limit_retries)
                print(f"⏳ Sleeping for {wait_time} seconds...")
                await asyncio.sleep(wait_time)
                print("🔄 Retrying after rate limit wait...")
                continue
            
            print(f"❌ Unexpected error: {error_str}")
            
            if attempts < max_rate_limit_retries:
                wait_time = 30 * attempts  # Simple linear backoff for other errors
                print(f"⏳ Waiting {wait_time} seconds before retry...")
                await asyncio.sleep(wait_time)
            else:
                print(f"💥 Max attempts reached. Stopping.")
                break
    
    return tweets

def analyze_data_quality(tweets_data, stage=""):
    """Phân tích chất lượng dữ liệu"""
    if not tweets_data:
        print(f"⚠️  {stage}: Không có dữ liệu để phân tích")
        return
    
    df = pd.DataFrame(tweets_data)
    
    print(f"\n📊 {stage} - PHÂN TÍCH DỮ LIỆU:")
    print("=" * 50)
    print(f"📈 Tổng số tweets: {len(df)}")
    print(f"👥 Số users unique: {df['user'].nunique()}")
    
    if 'text' in df.columns:
        text_lengths = df['text'].str.len()
        print(f"📝 Độ dài text - Trung bình: {text_lengths.mean():.1f}")
        print(f"📝 Độ dài text - Min: {text_lengths.min()}, Max: {text_lengths.max()}")
        
        # Hiển thị sample tweets
        print(f"\n🔍 Sample tweets:")
        for i, tweet in df.head(3).iterrows():
            text_preview = tweet['text'][:100] + "..." if len(tweet['text']) > 100 else tweet['text']
            print(f"  {i+1}. @{tweet['user']}: {text_preview}")

async def main():
    print_rate_limit_info()
    
    # Login
    client = Client(language='en-US')
    
    # Thử load cookies trước, nếu fail thì login lại
    login_success = False
    
    if os.path.exists('cookies.json'):
        print("\n🍪 Loading saved cookies...")
        try:
            client.load_cookies('cookies.json')
            print("✅ Cookies loaded successfully!")
            
            # Test cookies bằng cách thử một request đơn giản
            try:
                test_tweets = await client.search_tweet("test", count=1)
                print("✅ Cookies are valid!")
                login_success = True
            except Exception as e:
                print(f"❌ Cookies invalid: {str(e)}")
                print("🔄 Will attempt fresh login...")
                login_success = False
        except Exception as e:
            print(f"❌ Failed to load cookies: {str(e)}")
            login_success = False
    
    if not login_success:
        print("\n🔐 Performing fresh login...")
        try:
            await client.login(auth_info_1=username, auth_info_2=email, password=password)
            client.save_cookies('cookies.json')
            print("✅ Login successful, cookies saved!")
        except Exception as e:
            print(f"❌ Login failed: {str(e)}")
            return
    
    print(f"\n🎯 Bắt đầu thu thập {min_tweets} tweets với query: '{QUERY}'")
    
    # Thu thập tweets với rate limit handling
    tweets_data = await robust_search_tweets(
        client,
        QUERY,
        min_tweets,
        product=product,
        count_per_request=count_per_request,
        request_delay_seconds=request_delay_seconds,
        max_rate_limit_retries=max_rate_limit_retries,
    )
    
    if not tweets_data:
        print("❌ Không thu thập được tweets nào!")
        return
    
    # Phân tích dữ liệu trước khi clean
    analyze_data_quality(tweets_data, "TRƯỚC KHI CLEAN")
    
    # Clean dữ liệu
    print(f"\n🧹 Bắt đầu làm sạch dữ liệu...")
    for tweet in tweets_data:
        original_text = tweet['text']
        tweet['text'] = clean_tweet_text(original_text)
        tweet['text_length'] = len(tweet['text'])
    
    # Loại bỏ tweets có text quá ngắn sau khi clean (< 10 ký tự)
    tweets_data = [tweet for tweet in tweets_data if len(tweet['text']) >= 10]
    
    # Phân tích dữ liệu sau khi clean
    analyze_data_quality(tweets_data, "SAU KHI CLEAN")
    
    # So sánh before/after
    print(f"\n🔄 SO SÁNH BEFORE/AFTER CLEANING:")
    print("=" * 50)
    sample_tweet = tweets_data[0] if tweets_data else None
    if sample_tweet:
        print("📝 Ví dụ text sau khi clean:")
        print(f"   Length: {sample_tweet['text_length']} chars")
        print(f"   Content: {sample_tweet['text'][:200]}...")
    
    # Lưu vào CSV
    if tweets_data:
        df = pd.DataFrame(tweets_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bitcoin_tweets_cleaned_{timestamp}.csv"
        df.to_csv(filename, index=False, encoding='utf-8')
        
        print(f"\n ĐÃ LUU DỮ LIỆU:")
        print(f" File: {filename}")
        print(f" Số tweets: {len(df)}")
        print(f" Kích thước file: {os.path.getsize(filename)} bytes")
        
        print(f"\n HOÀN THÀNH! Thu thập được {len(tweets_data)} tweets Bitcoin")
        print(f"✨ Dữ liệu đã được làm sạch và lưu vào {filename}")
    else:
        print("❌ Không có dữ liệu để lưu!")

if __name__ == "__main__":
    asyncio.run(main())






