import os
import sys
from configparser import ConfigParser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from emoji_dict import clean_emoji_text, get_emoji_stats


def read_config_csv_path() -> str:
    config = ConfigParser()
    try:
        with open('config.ini', 'r', encoding='utf-8') as f:
            config.read_file(f)
    except FileNotFoundError:
        config.read('config.ini')
    if 'File' not in config or 'filename' not in config['File']:
        raise ValueError("config.ini thiếu [File].filename")
    filename = config['File']['filename'].strip()
    if not filename:
        raise ValueError("Giá trị filename trong config.ini rỗng")
    
    # Tìm file trong thư mục data/tweets/
    tweets_path = os.path.join('data', 'tweets', filename)
    if os.path.exists(tweets_path):
        return tweets_path
    
    # Nếu không tìm thấy trong data/tweets/, thử tìm ở thư mục gốc
    if os.path.exists(filename):
        return filename
    
    # Nếu không tìm thấy ở đâu, trả về đường dẫn trong data/tweets/ (sẽ báo lỗi sau)
    return tweets_path


def try_parse_datetime(series: pd.Series) -> pd.Series:
    """
    Parse datetime from various formats found in Twitter data
    """
    return pd.to_datetime(series.astype(str), errors='coerce', utc=True, format='%a %b %d %H:%M:%S %z %Y')


def _ensure_output_dir(dir_path: str) -> None:
    os.makedirs(dir_path, exist_ok=True)


def _init_plot_style() -> None:
    sns.set_theme(style='whitegrid')
        


def _safe_savefig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def _plot_time_series(df: pd.DataFrame, out_dir: str) -> None:
    if 'created_at' not in df.columns:
        return
    created = try_parse_datetime(df['created_at'])
    tmp = df.copy()
    tmp['created_dt'] = created
    tmp = tmp.dropna(subset=['created_dt'])
    if tmp.empty:
        return
    tmp['date'] = tmp['created_dt'].dt.date
    daily_counts = tmp.groupby('date').size().reset_index(name='num_tweets')

    _init_plot_style()
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=daily_counts, x='date', y='num_tweets', marker='o')
    plt.title('Số lượng tweet theo ngày')
    plt.xlabel('Ngày')
    plt.ylabel('Số tweet')
    plt.xticks(rotation=30, ha='right')
    _safe_savefig(os.path.join(out_dir, 'time_series_daily.png'))

    tmp['hour'] = tmp['created_dt'].dt.floor('h')
    hourly_counts = tmp.groupby('hour').size().reset_index(name='num_tweets')
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=hourly_counts, x='hour', y='num_tweets')
    plt.title('Số lượng tweet theo giờ')
    plt.xlabel('Giờ')
    plt.ylabel('Số tweet')
    plt.xticks(rotation=30, ha='right')
    _safe_savefig(os.path.join(out_dir, 'time_series_hourly.png'))


def _plot_text_length(df: pd.DataFrame, out_dir: str) -> None:
    if 'text' not in df.columns:
        return
    text_len = df['text'].astype(str).str.len()
    _init_plot_style()
    plt.figure(figsize=(8, 4))
    sns.histplot(text_len, bins=40, kde=True)
    plt.title('Phân phối độ dài văn bản tweet')
    plt.xlabel('Số ký tự')
    plt.ylabel('Tần suất')
    _safe_savefig(os.path.join(out_dir, 'text_length_distribution.png'))


def _plot_top_users(df: pd.DataFrame, out_dir: str, top_n: int = 20) -> None:
    if 'user' not in df.columns:
        return
    counts = df['user'].value_counts().head(top_n)
    if counts.empty:
        return
    _init_plot_style()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts.values, y=counts.index, orient='h')
    plt.title(f'Top {top_n} người dùng theo số tweet')
    plt.xlabel('Số tweet')
    plt.ylabel('User')
    _safe_savefig(os.path.join(out_dir, 'top_users.png'))


def _plot_engagement(df: pd.DataFrame, out_dir: str) -> None:
    cols = [c for c in ['retweet_count', 'favorite_count', 'reply_count'] if c in df.columns]
    if not cols:
        return
    melted = df[cols].melt(var_name='metric', value_name='value')
    _init_plot_style()
    plt.figure(figsize=(10, 4))
    sns.boxplot(data=melted, x='metric', y='value')
    sns.stripplot(data=melted, x='metric', y='value', color='black', size=2, alpha=0.2)
    plt.title('Phân phối tương tác (retweet/favorite/reply)')
    plt.xlabel('Chỉ số')
    plt.ylabel('Giá trị')
    _safe_savefig(os.path.join(out_dir, 'engagement_distribution.png'))


def _generate_charts(df: pd.DataFrame, source_file: str) -> None:
    base = os.path.splitext(os.path.basename(source_file))[0]
    out_dir = os.path.join('charts', base)
    _ensure_output_dir(out_dir)

    print(f"\nĐang lưu biểu đồ vào thư mục: {out_dir}")
    _plot_time_series(df, out_dir)
    _plot_text_length(df, out_dir)
    _plot_top_users(df, out_dir)
    _plot_engagement(df, out_dir)


def _summarize_and_conclude(df: pd.DataFrame, source_file: str) -> None:
    base = os.path.splitext(os.path.basename(source_file))[0]
    out_dir = os.path.join('charts', base)
    _ensure_output_dir(out_dir)

    lines = []

    # Basic overview
    total = len(df)
    lines.append(f"Tệp dữ liệu: {source_file}")
    lines.append(f"Tổng số tweet: {total}")

    # Time-based insights
    if 'created_at' in df.columns:
        created = try_parse_datetime(df['created_at'])
        dt = created.dropna()
        if not dt.empty:
            start, end = dt.min(), dt.max()
            lines.append(f"Khoảng thời gian: {start} -> {end}")

            daily = dt.dt.date.value_counts().sort_index()
            if not daily.empty:
                avg_per_day = float(np.mean(daily.values))
                peak_day = daily.idxmax()
                peak_day_count = int(daily.max())
                # Trend via simple linear regression on index
                x = np.arange(len(daily))
                y = daily.values.astype(float)
                slope = float(np.polyfit(x, y, 1)[0]) if len(daily) >= 2 else 0.0
                trend = 'tăng' if slope > 0 else ('giảm' if slope < 0 else 'ổn định')
                lines.append(f"Trung bình {avg_per_day:.1f} tweet/ngày; ngày cao điểm: {peak_day} với {peak_day_count} tweet; xu hướng chung: {trend}.")

            hourly = dt.dt.floor('h').value_counts().sort_index()
            if not hourly.empty:
                peak_hour = str(hourly.idxmax())
                peak_hour_count = int(hourly.max())
                lines.append(f"Khung giờ sôi động nhất: {peak_hour} với {peak_hour_count} tweet.")

    # Text insights
    if 'text' in df.columns:
        text_len = df['text'].astype(str).str.len()
        if not text_len.empty:
            mean_len = float(text_len.mean())
            median_len = float(text_len.median())
            p95_len = int(np.percentile(text_len, 95))
            lines.append(f"Độ dài văn bản: mean {mean_len:.1f}, median {median_len:.1f}, 95th pct {p95_len} ký tự.")

    # Users
    if 'user' in df.columns:
        nunique_users = int(df['user'].nunique())
        top_users = df['user'].value_counts().head(5)
        lines.append(f"Số lượng người dùng duy nhất: {nunique_users}.")
        if not top_users.empty:
            top_str = ", ".join([f"{idx} ({val})" for idx, val in top_users.items()])
            lines.append(f"Top 5 người dùng hoạt động: {top_str}.")

    # Engagement
    engagement_cols = [c for c in ['retweet_count', 'favorite_count', 'reply_count'] if c in df.columns]
    if engagement_cols:
        medians = df[engagement_cols].median(numeric_only=True)
        p90 = df[engagement_cols].quantile(0.9, numeric_only=True)
        parts = []
        for col in engagement_cols:
            parts.append(f"{col}: median {int(medians[col]) if pd.notna(medians[col]) else 'NA'}, p90 {int(p90[col]) if pd.notna(p90[col]) else 'NA'}")
        lines.append("Tương tác (trung vị/p90): " + "; ".join(parts) + ".")

    # Conclusion heuristic
    conclusion = []
    if 'created_at' in df.columns:
        created = try_parse_datetime(df['created_at'])
        dt = created.dropna()
        if not dt.empty:
            daily = dt.dt.date.value_counts().sort_index()
            if len(daily) >= 2:
                slope = float(np.polyfit(np.arange(len(daily)), daily.values.astype(float), 1)[0])
                if slope > 0:
                    conclusion.append("Khối lượng thảo luận đang có xu hướng tăng, gợi ý mối quan tâm ngày càng cao.")
                elif slope < 0:
                    conclusion.append("Khối lượng thảo luận giảm dần, sự chú ý có thể hạ nhiệt.")
                else:
                    conclusion.append("Khối lượng thảo luận ổn định trong giai đoạn quan sát.")
    if engagement_cols:
        med = df[engagement_cols].median(numeric_only=True)
        if (med.fillna(0) > 0).any():
            conclusion.append("Mức tương tác trung vị > 0 cho thấy có phản hồi thực sự từ cộng đồng.")
        else:
            conclusion.append("Tương tác trung vị thấp; nội dung phần lớn có độ lan truyền hạn chế.")

    if not conclusion:
        conclusion.append("Dữ liệu cho thấy các đặc điểm ổn định; cần theo dõi thêm để xác nhận xu hướng.")

    # Print to console
    print("\n===== PHÂN TÍCH & GIẢI THÍCH =====")
    for line in lines:
        print("- " + line)
    print("\n===== KẾT LUẬN =====")
    for c in conclusion:
        print("• " + c)

    # Save report
    report_path = os.path.join(out_dir, 'report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Báo cáo phân tích Twitter\n\n")
        f.write("## Giải thích\n")
        for line in lines:
            f.write(f"- {line}\n")
        f.write("\n## Kết luận\n")
        for c in conclusion:
            f.write(f"- {c}\n")
    print(f"\nĐã lưu báo cáo: {report_path}")


def analyze_csv(file_path: str) -> None:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    df = pd.read_csv(file_path, encoding='utf-8')

    print("\n===== TỔNG QUAN DỮ LIỆU =====")
    print(f"File: {file_path}")
    print(f"Số dòng, cột: {df.shape}")
    print(f"Các cột: {list(df.columns)}")

    # Null overview
    null_counts = df.isna().sum()
    non_zero_nulls = null_counts[null_counts > 0]
    if len(non_zero_nulls) > 0:
        print("\nCột có giá trị null:")
        print(non_zero_nulls.sort_values(ascending=False))
    else:
        print("\nKhông có giá trị null.")

    # Time range
    if 'created_at' in df.columns:
        created = try_parse_datetime(df['created_at'])
        print("\n===== THỜI GIAN =====")
        print(f"Khoảng thời gian: {created.min()}  ->  {created.max()}")
    else:
        print("\nKhông có cột 'created_at' để phân tích thời gian.")

    # Users
    if 'user' in df.columns:
        print("\n===== NGƯỜI DÙNG =====")
        print(f"Số user duy nhất: {df['user'].nunique()}")
        top_users = df['user'].value_counts().head(10)
        print("Top 10 user theo số tweet:")
        print(top_users)

    # Text stats
    if 'text' in df.columns:
        print("\n===== VĂN BẢN =====")
        text_len = df['text'].astype(str).str.len()
        print(f"Độ dài text - mean: {text_len.mean():.1f}, min: {text_len.min()}, max: {text_len.max()}")
        long_examples = df.loc[text_len.nlargest(3).index, ['user', 'text']]
        print("\n3 ví dụ text dài nhất (cắt 200 ký tự):")
        for _, row in long_examples.iterrows():
            content = (row['text'][:200] + '...') if len(row['text']) > 200 else row['text']
            print(f"@{row.get('user', 'unknown')}: {content}")

    # Engagement
    engagement_cols = [c for c in ['retweet_count', 'favorite_count', 'reply_count'] if c in df.columns]
    if engagement_cols:
        print("\n===== TƯƠNG TÁC =====")
        stats = df[engagement_cols].describe().T[['mean', 'std', 'min', '50%', 'max']]
        print(stats)

    # Sample
    print("\n===== MẪU DỮ LIỆU (5 dòng đầu) =====")
    with pd.option_context('display.max_colwidth', 120):
        print(df.head(5))

    # Charts
    _generate_charts(df, file_path)

    # Narrative summary
    _summarize_and_conclude(df, file_path)


if __name__ == "__main__":
    try:
        csv_file = read_config_csv_path()
        analyze_csv(csv_file)
    except Exception as e:
        print(f"Lỗi: {e}")
        sys.exit(1) 