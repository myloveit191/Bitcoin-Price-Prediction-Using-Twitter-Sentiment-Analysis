import asyncio
from twikit import Client, TooManyRequests
import time
from datetime import datetime, timedelta
import pandas as pd
import re
import html
from configparser import ConfigParser
from random import randint
import os
import json

# Load config - sử dụng cách giống scraping.py
config = ConfigParser()
try:
    with open('config.ini', 'r', encoding='utf-8') as f:
        config.read_file(f)
except FileNotFoundError:
    config.read('config.ini')  # fallback
    
username = config['X']['username']
password = config['X']['password']
email = config['X']['email']

# Read scraping configuration - copy từ scraping.py
scraping_section = 'Scraping'
min_tweets = config.getint(scraping_section, 'min_tweets', fallback=5000)
start_date_str = config.get(scraping_section, 'start_date', fallback='2025-08-15')
end_date_str = config.get(scraping_section, 'end_date', fallback='2025-09-15')
keywords = config.get(scraping_section, 'keywords', fallback='bitcoin')
lang = config.get(scraping_section, 'lang', fallback='en')
provided_query = config.get(scraping_section, 'query', fallback='').strip()
product = config.get(scraping_section, 'product', fallback='Top')
count_per_request = config.getint(scraping_section, 'count_per_request', fallback=20)
tweets_per_day = 200  # Target tweets per day
request_delay_seconds = config.getint(scraping_section, 'request_delay_seconds', fallback=10)
max_rate_limit_retries = config.getint(scraping_section, 'max_rate_limit_retries', fallback=5)

def clean_tweet_text(text):
    """Clean tweet text"""
    if not text:
        return ""
    
    # HTML decode
    text = html.unescape(text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def generate_date_ranges(start_date_str, end_date_str):
    """Generate list of date ranges for daily scraping"""
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    
    date_ranges = []
    current_date = start_date
    
    while current_date <= end_date:
        next_date = current_date + timedelta(days=1)
        date_ranges.append({
            'start': current_date.strftime('%Y-%m-%d'),
            'end': next_date.strftime('%Y-%m-%d'),
            'display': current_date.strftime('%Y-%m-%d')
        })
        current_date = next_date
    
    return date_ranges

async def scrape_tweets_for_date(client, date_range, tweets_per_day=200):
    """Scrape tweets for a specific date"""
    query = f"{keywords} lang:{lang} since:{date_range['start']} until:{date_range['end']}"
    
    print(f"\n🔍 Thu thập tweets cho ngày: {date_range['display']}")
    print(f"📝 Query: {query}")
    
    tweets_data = []
    cursor = None
    retries = 0
    
    try:
        while len(tweets_data) < tweets_per_day:
            try:
                if cursor:
                    tweets = await client.search_tweet(query, product=product, cursor=cursor)
                else:
                    tweets = await client.search_tweet(query, product=product)
                
                if not tweets:
                    print(f"   ⚠️ Không tìm thấy thêm tweets cho {date_range['display']}")
                    break
                
                # Process tweets
                batch_count = 0
                for tweet in tweets:
                    if len(tweets_data) >= tweets_per_day:
                        break
                    
                    tweet_data = {
                        'id': tweet.id,
                        'text': tweet.text or '',
                        'user': tweet.user.screen_name if tweet.user else 'unknown',
                        'created_at': tweet.created_at or '',
                        'retweet_count': getattr(tweet, 'retweet_count', 0),
                        'favorite_count': getattr(tweet, 'favorite_count', 0),
                        'reply_count': getattr(tweet, 'reply_count', 0),
                        'date': date_range['display']
                    }
                    
                    tweets_data.append(tweet_data)
                    batch_count += 1
                
                print(f"   ✅ Thu thập được {batch_count} tweets (Tổng: {len(tweets_data)}/{tweets_per_day})")
                
                # Get cursor for next page
                cursor = tweets.next_cursor if hasattr(tweets, 'next_cursor') else None
                if not cursor:
                    break
                
                # Delay between requests
                delay = randint(request_delay_seconds, request_delay_seconds + 5)
                print(f"   ⏳ Đợi {delay} giây...")
                await asyncio.sleep(delay)
                
            except TooManyRequests as e:
                retries += 1
                if retries > max_rate_limit_retries:
                    print(f"   ❌ Vượt quá giới hạn retry cho {date_range['display']}")
                    break
                
                reset_time = e.reset_time if hasattr(e, 'reset_time') else 900
                print(f"   ⏰ Rate limit hit. Đợi {reset_time} giây...")
                await asyncio.sleep(reset_time)
                
            except Exception as e:
                print(f"   ❌ Lỗi khi thu thập {date_range['display']}: {e}")
                break
                
    except Exception as e:
        print(f"❌ Lỗi nghiêm trọng cho {date_range['display']}: {e}")
    
    return tweets_data

async def main():
    """Main function for daily Twitter scraping"""
    print("🚀 BẮT ĐẦU THU THẬP TWITTER DATA THEO NGÀY")
    print("=" * 60)
    print(f"📅 Khoảng thời gian: {start_date_str} đến {end_date_str}")
    print(f"🎯 Mục tiêu: {tweets_per_day} tweets/ngày")
    print(f"🔍 Keywords: {keywords}")
    print("=" * 60)
    
    # Initialize client
    client = Client(language='en-US')
    
    # Login
    try:
        if os.path.exists('cookies.json'):
            print("🍪 Đang load cookies...")
            client.load_cookies('cookies.json')
        else:
            print("🔐 Đang đăng nhập Twitter...")
            await client.login(auth_info_1=username, auth_info_2=email, password=password)
            client.save_cookies('cookies.json')
            print("✅ Đăng nhập thành công!")
    except Exception as e:
        print(f"❌ Lỗi đăng nhập: {e}")
        return
    
    # Generate date ranges
    date_ranges = generate_date_ranges(start_date_str, end_date_str)
    total_days = len(date_ranges)
    
    print(f"\n📋 Sẽ thu thập {total_days} ngày")
    
    all_tweets = []
    successful_days = 0
    
    # Scrape each day
    for i, date_range in enumerate(date_ranges, 1):
        print(f"\n{'='*50}")
        print(f"📅 NGÀY {i}/{total_days}: {date_range['display']}")
        print(f"{'='*50}")
        
        daily_tweets = await scrape_tweets_for_date(client, date_range, tweets_per_day)
        
        if daily_tweets:
            # Clean tweets
            for tweet in daily_tweets:
                tweet['text'] = clean_tweet_text(tweet['text'])
                tweet['text_length'] = len(tweet['text'])
            
            # Filter out very short tweets
            daily_tweets = [t for t in daily_tweets if len(t['text']) >= 10]
            
            all_tweets.extend(daily_tweets)
            successful_days += 1
            
            print(f"✅ Ngày {date_range['display']}: {len(daily_tweets)} tweets (sau khi clean)")
            
            # Save daily backup
            if daily_tweets:
                df_daily = pd.DataFrame(daily_tweets)
                # Ensure data/tweets directory exists
                os.makedirs("data/tweets", exist_ok=True)
                daily_filename = f"data/tweets/bitcoin_tweets_{date_range['display']}.csv"
                df_daily.to_csv(daily_filename, index=False, encoding='utf-8')
                print(f"💾 Đã lưu backup: {daily_filename}")
        else:
            print(f"❌ Ngày {date_range['display']}: Không thu thập được tweets")
        
        # Delay between days
        if i < total_days:
            delay = randint(30, 60)
            print(f"⏳ Nghỉ {delay} giây trước khi chuyển sang ngày tiếp theo...")
            await asyncio.sleep(delay)
    
    # Save final combined file
    if all_tweets:
        print(f"\n🎉 HOÀN THÀNH THU THẬP!")
        print("=" * 60)
        print(f"📊 Tổng kết:")
        print(f"   - Ngày thành công: {successful_days}/{total_days}")
        print(f"   - Tổng tweets: {len(all_tweets)}")
        print(f"   - Trung bình: {len(all_tweets)/successful_days:.1f} tweets/ngày" if successful_days > 0 else "")
        
        # Create final DataFrame
        df_final = pd.DataFrame(all_tweets)
        
        # Add summary stats
        print(f"\n📈 Thống kê:")
        print(f"   - Unique users: {df_final['user'].nunique()}")
        print(f"   - Avg text length: {df_final['text_length'].mean():.1f} chars")
        print(f"   - Date range: {df_final['date'].min()} to {df_final['date'].max()}")
        
        # Save final file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Ensure data/tweets directory exists
        os.makedirs("data/tweets", exist_ok=True)
        final_filename = f"data/tweets/bitcoin_tweets_1month_{timestamp}.csv"
        df_final.to_csv(final_filename, index=False, encoding='utf-8')
        
        print(f"\n💾 ĐÃ LƯU FILE CUỐI CÙNG: {final_filename}")
        print(f"📁 Kích thước: {os.path.getsize(final_filename)/1024/1024:.2f} MB")
        
        # Show sample
        print(f"\n📝 Mẫu dữ liệu:")
        print(df_final[['date', 'user', 'text_length', 'retweet_count']].head())
        
    else:
        print("❌ Không thu thập được tweets nào!")

if __name__ == "__main__":
    asyncio.run(main()) 