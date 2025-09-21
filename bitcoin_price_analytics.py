import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Thiết lập style cho biểu đồ
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BitcoinPriceAnalyzer:
    def __init__(self, data_path: str):
        """
        Khởi tạo analyzer cho dữ liệu Bitcoin price
        
        Args:
            data_path: Đường dẫn đến file CSV chứa dữ liệu Bitcoin price
        """
        self.data_path = data_path
        self.df = None
        self.output_dir = None
        
    def load_data(self):
        """Load và tiền xử lý dữ liệu Bitcoin price"""
        print("Đang tải dữ liệu Bitcoin price...")
        
        # Load dữ liệu
        self.df = pd.read_csv(self.data_path)
        
        # Chuyển đổi timestamp
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['fetch_timestamp'] = pd.to_datetime(self.df['fetch_timestamp'])
        
        # Sắp xếp theo thời gian
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)
        
        # Tạo các cột thời gian bổ sung
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['day_of_week'] = self.df['timestamp'].dt.day_name()
        self.df['month'] = self.df['timestamp'].dt.month
        
        # Tính toán các chỉ số kỹ thuật
        self._calculate_technical_indicators()
        
        print(f"Đã tải {len(self.df)} bản ghi từ {self.df['timestamp'].min()} đến {self.df['timestamp'].max()}")
        
    def _calculate_technical_indicators(self):
        """Tính toán các chỉ số kỹ thuật"""
        # Price changes
        self.df['price_change'] = self.df['price_usd'].diff()
        self.df['price_change_pct'] = self.df['price_usd'].pct_change() * 100
        
        # Moving averages
        self.df['ma_24h'] = self.df['price_usd'].rolling(window=24, min_periods=1).mean()
        self.df['ma_7d'] = self.df['price_usd'].rolling(window=168, min_periods=1).mean()  # 7 days * 24 hours
        
        # Volatility (rolling standard deviation)
        self.df['volatility_24h'] = self.df['price_usd'].rolling(window=24, min_periods=1).std()
        
        # Volume metrics
        self.df['volume_change'] = self.df['volume_usd'].diff()
        self.df['volume_change_pct'] = self.df['volume_usd'].pct_change() * 100
        
        # Market cap change
        self.df['market_cap_change'] = self.df['market_cap_usd'].diff()
        self.df['market_cap_change_pct'] = self.df['market_cap_usd'].pct_change() * 100
        
    def create_output_directory(self):
        """Tạo thư mục output cho biểu đồ và báo cáo"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"charts/bitcoin_price_analysis_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Thư mục output: {self.output_dir}")
        
    def plot_price_timeline(self):
        """Biểu đồ giá Bitcoin theo thời gian"""
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Biểu đồ giá chính
        axes[0].plot(self.df['timestamp'], self.df['price_usd'], linewidth=1.5, alpha=0.8, label='Giá Bitcoin')
        axes[0].plot(self.df['timestamp'], self.df['ma_24h'], linewidth=2, alpha=0.7, label='MA 24h', color='orange')
        axes[0].plot(self.df['timestamp'], self.df['ma_7d'], linewidth=2, alpha=0.7, label='MA 7 ngày', color='red')
        
        axes[0].set_title('Giá Bitcoin USD theo thời gian', fontsize=16, fontweight='bold')
        axes[0].set_ylabel('Giá (USD)', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Biểu đồ volume
        axes[1].bar(self.df['timestamp'], self.df['volume_usd'] / 1e9, alpha=0.6, width=0.8)
        axes[1].set_title('Volume giao dịch (tỷ USD)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Volume (tỷ USD)', fontsize=12)
        axes[1].set_xlabel('Thời gian', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'price_timeline.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_price_distribution(self):
        """Phân phối giá Bitcoin"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Histogram giá
        axes[0, 0].hist(self.df['price_usd'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Phân phối giá Bitcoin', fontweight='bold')
        axes[0, 0].set_xlabel('Giá (USD)')
        axes[0, 0].set_ylabel('Tần suất')
        axes[0, 0].axvline(self.df['price_usd'].mean(), color='red', linestyle='--', label=f'Mean: ${self.df["price_usd"].mean():,.0f}')
        axes[0, 0].axvline(self.df['price_usd'].median(), color='green', linestyle='--', label=f'Median: ${self.df["price_usd"].median():,.0f}')
        axes[0, 0].legend()
        
        # Box plot giá
        axes[0, 1].boxplot(self.df['price_usd'], patch_artist=True)
        axes[0, 1].set_title('Box Plot giá Bitcoin', fontweight='bold')
        axes[0, 1].set_ylabel('Giá (USD)')
        
        # Phân phối thay đổi giá
        axes[1, 0].hist(self.df['price_change_pct'].dropna(), bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Phân phối thay đổi giá (%)', fontweight='bold')
        axes[1, 0].set_xlabel('Thay đổi giá (%)')
        axes[1, 0].set_ylabel('Tần suất')
        axes[1, 0].axvline(0, color='red', linestyle='--', alpha=0.7)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(self.df['price_usd'], dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Normal Distribution)', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'price_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_volume_analysis(self):
        """Phân tích volume giao dịch"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Volume theo thời gian
        axes[0, 0].plot(self.df['timestamp'], self.df['volume_usd'] / 1e9, alpha=0.7)
        axes[0, 0].set_title('Volume giao dịch theo thời gian', fontweight='bold')
        axes[0, 0].set_ylabel('Volume (tỷ USD)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Volume vs Price scatter
        axes[0, 1].scatter(self.df['volume_usd'] / 1e9, self.df['price_usd'], alpha=0.6)
        axes[0, 1].set_title('Mối quan hệ Volume vs Giá', fontweight='bold')
        axes[0, 1].set_xlabel('Volume (tỷ USD)')
        axes[0, 1].set_ylabel('Giá (USD)')
        
        # Volume theo giờ trong ngày
        hourly_volume = self.df.groupby('hour')['volume_usd'].mean() / 1e9
        axes[1, 0].bar(hourly_volume.index, hourly_volume.values, alpha=0.7)
        axes[1, 0].set_title('Volume trung bình theo giờ', fontweight='bold')
        axes[1, 0].set_xlabel('Giờ trong ngày')
        axes[1, 0].set_ylabel('Volume TB (tỷ USD)')
        
        # Volume theo ngày trong tuần
        daily_volume = self.df.groupby('day_of_week')['volume_usd'].mean() / 1e9
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_volume = daily_volume.reindex(day_order)
        axes[1, 1].bar(range(len(daily_volume)), daily_volume.values, alpha=0.7)
        axes[1, 1].set_title('Volume trung bình theo ngày', fontweight='bold')
        axes[1, 1].set_xlabel('Ngày trong tuần')
        axes[1, 1].set_ylabel('Volume TB (tỷ USD)')
        axes[1, 1].set_xticks(range(len(day_order)))
        axes[1, 1].set_xticklabels([d[:3] for d in day_order], rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'volume_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_volatility_analysis(self):
        """Phân tích độ biến động"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Volatility theo thời gian
        axes[0, 0].plot(self.df['timestamp'], self.df['volatility_24h'], alpha=0.7, color='red')
        axes[0, 0].set_title('Độ biến động 24h theo thời gian', fontweight='bold')
        axes[0, 0].set_ylabel('Volatility (USD)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Volatility vs Price
        axes[0, 1].scatter(self.df['price_usd'], self.df['volatility_24h'], alpha=0.6)
        axes[0, 1].set_title('Volatility vs Giá', fontweight='bold')
        axes[0, 1].set_xlabel('Giá (USD)')
        axes[0, 1].set_ylabel('Volatility (USD)')
        
        # Volatility theo giờ
        hourly_vol = self.df.groupby('hour')['volatility_24h'].mean()
        axes[1, 0].bar(hourly_vol.index, hourly_vol.values, alpha=0.7, color='orange')
        axes[1, 0].set_title('Volatility trung bình theo giờ', fontweight='bold')
        axes[1, 0].set_xlabel('Giờ trong ngày')
        axes[1, 0].set_ylabel('Volatility TB (USD)')
        
        # Rolling correlation giữa price và volume
        rolling_corr = self.df['price_usd'].rolling(window=24).corr(self.df['volume_usd'])
        axes[1, 1].plot(self.df['timestamp'], rolling_corr, alpha=0.7, color='green')
        axes[1, 1].set_title('Correlation Price-Volume (24h rolling)', fontweight='bold')
        axes[1, 1].set_ylabel('Correlation')
        axes[1, 1].axhline(0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'volatility_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_market_cap_analysis(self):
        """Phân tích market cap"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Market cap theo thời gian
        axes[0, 0].plot(self.df['timestamp'], self.df['market_cap_usd'] / 1e12, linewidth=2)
        axes[0, 0].set_title('Market Cap theo thời gian', fontweight='bold')
        axes[0, 0].set_ylabel('Market Cap (nghìn tỷ USD)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Market cap vs Price
        axes[0, 1].scatter(self.df['price_usd'], self.df['market_cap_usd'] / 1e12, alpha=0.6)
        axes[0, 1].set_title('Market Cap vs Giá', fontweight='bold')
        axes[0, 1].set_xlabel('Giá (USD)')
        axes[0, 1].set_ylabel('Market Cap (nghìn tỷ USD)')
        
        # Thay đổi market cap
        axes[1, 0].plot(self.df['timestamp'], self.df['market_cap_change_pct'], alpha=0.7, color='purple')
        axes[1, 0].set_title('Thay đổi Market Cap (%)', fontweight='bold')
        axes[1, 0].set_ylabel('Thay đổi (%)')
        axes[1, 0].axhline(0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Distribution of market cap changes
        axes[1, 1].hist(self.df['market_cap_change_pct'].dropna(), bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Phân phối thay đổi Market Cap', fontweight='bold')
        axes[1, 1].set_xlabel('Thay đổi (%)')
        axes[1, 1].set_ylabel('Tần suất')
        axes[1, 1].axvline(0, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'market_cap_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_summary_statistics(self):
        """Tạo thống kê tổng quan"""
        stats = {}
        
        # Basic price statistics
        stats['price'] = {
            'min': self.df['price_usd'].min(),
            'max': self.df['price_usd'].max(),
            'mean': self.df['price_usd'].mean(),
            'median': self.df['price_usd'].median(),
            'std': self.df['price_usd'].std(),
            'q25': self.df['price_usd'].quantile(0.25),
            'q75': self.df['price_usd'].quantile(0.75)
        }
        
        # Price changes
        price_changes = self.df['price_change_pct'].dropna()
        stats['price_changes'] = {
            'mean_daily_change': price_changes.mean(),
            'std_daily_change': price_changes.std(),
            'max_gain': price_changes.max(),
            'max_loss': price_changes.min(),
            'positive_days': (price_changes > 0).sum(),
            'negative_days': (price_changes < 0).sum()
        }
        
        # Volume statistics
        stats['volume'] = {
            'mean': self.df['volume_usd'].mean(),
            'median': self.df['volume_usd'].median(),
            'max': self.df['volume_usd'].max(),
            'min': self.df['volume_usd'].min()
        }
        
        # Market cap statistics
        stats['market_cap'] = {
            'mean': self.df['market_cap_usd'].mean(),
            'median': self.df['market_cap_usd'].median(),
            'max': self.df['market_cap_usd'].max(),
            'min': self.df['market_cap_usd'].min()
        }
        
        # Volatility
        stats['volatility'] = {
            'mean_24h': self.df['volatility_24h'].mean(),
            'max_24h': self.df['volatility_24h'].max(),
            'min_24h': self.df['volatility_24h'].min()
        }
        
        return stats
        
    def generate_report(self):
        """Tạo báo cáo tổng hợp"""
        stats = self.generate_summary_statistics()
        
        # Tạo báo cáo markdown
        report_content = f"""# Báo cáo phân tích giá Bitcoin

## Tổng quan dữ liệu
- **Khoảng thời gian**: {self.df['timestamp'].min().strftime('%Y-%m-%d %H:%M')} đến {self.df['timestamp'].max().strftime('%Y-%m-%d %H:%M')}
- **Số bản ghi**: {len(self.df):,}
- **Tần suất**: Mỗi giờ
- **Nguồn dữ liệu**: {self.df['source'].iloc[0]}

## Thống kê giá Bitcoin

### Giá cơ bản
- **Giá thấp nhất**: ${stats['price']['min']:,.2f}
- **Giá cao nhất**: ${stats['price']['max']:,.2f}
- **Giá trung bình**: ${stats['price']['mean']:,.2f}
- **Giá trung vị**: ${stats['price']['median']:,.2f}
- **Độ lệch chuẩn**: ${stats['price']['std']:,.2f}

### Phân vị giá
- **Q25 (25%)**: ${stats['price']['q25']:,.2f}
- **Q75 (75%)**: ${stats['price']['q75']:,.2f}

### Thay đổi giá
- **Thay đổi trung bình/ngày**: {stats['price_changes']['mean_daily_change']:.2f}%
- **Độ biến động**: {stats['price_changes']['std_daily_change']:.2f}%
- **Tăng giá mạnh nhất**: {stats['price_changes']['max_gain']:.2f}%
- **Giảm giá mạnh nhất**: {stats['price_changes']['max_loss']:.2f}%
- **Số ngày tăng**: {stats['price_changes']['positive_days']}
- **Số ngày giảm**: {stats['price_changes']['negative_days']}

## Thống kê Volume

- **Volume trung bình**: ${stats['volume']['mean']:,.0f}
- **Volume trung vị**: ${stats['volume']['median']:,.0f}
- **Volume cao nhất**: ${stats['volume']['max']:,.0f}
- **Volume thấp nhất**: ${stats['volume']['min']:,.0f}

## Thống kê Market Cap

- **Market Cap trung bình**: ${stats['market_cap']['mean']:,.0f}
- **Market Cap trung vị**: ${stats['market_cap']['median']:,.0f}
- **Market Cap cao nhất**: ${stats['market_cap']['max']:,.0f}
- **Market Cap thấp nhất**: ${stats['market_cap']['min']:,.0f}

## Phân tích độ biến động

- **Volatility trung bình 24h**: ${stats['volatility']['mean_24h']:,.2f}
- **Volatility cao nhất**: ${stats['volatility']['max_24h']:,.2f}
- **Volatility thấp nhất**: ${stats['volatility']['min_24h']:,.2f}

## Biểu đồ được tạo

1. **price_timeline.png**: Biểu đồ giá và volume theo thời gian
2. **price_distribution.png**: Phân phối giá và các chỉ số thống kê
3. **volume_analysis.png**: Phân tích volume giao dịch
4. **volatility_analysis.png**: Phân tích độ biến động
5. **market_cap_analysis.png**: Phân tích market cap

## Kết luận

Dựa trên phân tích dữ liệu từ {self.df['timestamp'].min().strftime('%d/%m/%Y')} đến {self.df['timestamp'].max().strftime('%d/%m/%Y')}:

- Bitcoin có giá dao động từ ${stats['price']['min']:,.0f} đến ${stats['price']['max']:,.0f}
- Thay đổi giá trung bình hàng ngày là {stats['price_changes']['mean_daily_change']:.2f}%
- Tỷ lệ ngày tăng/giảm: {stats['price_changes']['positive_days']}/{stats['price_changes']['negative_days']}
- Độ biến động trung bình: ${stats['volatility']['mean_24h']:,.2f}

---
*Báo cáo được tạo tự động vào {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}*
"""
        
        # Lưu báo cáo
        report_path = os.path.join(self.output_dir, 'bitcoin_analysis_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
            
        print(f"Đã tạo báo cáo: {report_path}")
        
        # In thống kê ra console
        print("\n" + "="*50)
        print("THỐNG KÊ TỔNG QUAN BITCOIN")
        print("="*50)
        print(f"Khoảng thời gian: {self.df['timestamp'].min().strftime('%d/%m/%Y %H:%M')} - {self.df['timestamp'].max().strftime('%d/%m/%Y %H:%M')}")
        print(f"Số bản ghi: {len(self.df):,}")
        print(f"\nGiá Bitcoin:")
        print(f"  - Thấp nhất: ${stats['price']['min']:,.2f}")
        print(f"  - Cao nhất: ${stats['price']['max']:,.2f}")
        print(f"  - Trung bình: ${stats['price']['mean']:,.2f}")
        print(f"  - Trung vị: ${stats['price']['median']:,.2f}")
        print(f"\nThay đổi giá:")
        print(f"  - Trung bình/ngày: {stats['price_changes']['mean_daily_change']:.2f}%")
        print(f"  - Tăng mạnh nhất: {stats['price_changes']['max_gain']:.2f}%")
        print(f"  - Giảm mạnh nhất: {stats['price_changes']['max_loss']:.2f}%")
        print(f"  - Ngày tăng: {stats['price_changes']['positive_days']}")
        print(f"  - Ngày giảm: {stats['price_changes']['negative_days']}")
        
    def run_analysis(self):
        """Chạy toàn bộ phân tích"""
        print("Bắt đầu phân tích dữ liệu Bitcoin price...")
        
        # Load dữ liệu
        self.load_data()
        
        # Tạo thư mục output
        self.create_output_directory()
        
        # Tạo các biểu đồ
        print("Đang tạo biểu đồ...")
        self.plot_price_timeline()
        self.plot_price_distribution()
        self.plot_volume_analysis()
        self.plot_volatility_analysis()
        self.plot_market_cap_analysis()
        
        # Tạo báo cáo
        print("Đang tạo báo cáo...")
        self.generate_report()
        
        print(f"\nPhân tích hoàn tất! Kết quả được lưu trong: {self.output_dir}")


def main():
    """Hàm main để chạy phân tích"""
    # Đường dẫn đến file dữ liệu Bitcoin
    data_path = "data/bitcoin-price/bitcoin_price_data_20250919_165524.csv"
    
    if not os.path.exists(data_path):
        print(f"Không tìm thấy file dữ liệu: {data_path}")
        sys.exit(1)
    
    # Tạo analyzer và chạy phân tích
    analyzer = BitcoinPriceAnalyzer(data_path)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()