# Bitcoin Price Prediction Using Twitter Sentiment Analysis

## 1. Giới thiệu

Dự án nhằm kiểm tra xem **dữ liệu Twitter** có thể được dùng để dự đoán
**biến động giá Bitcoin** hay không.\
Phương pháp: 

## 2. Dữ liệu sử dụng
- Dữ liệu Twitter: Thu thập tweets về Bitcoin (keyword: "bitcoin")
- Dữ liệu giá Bitcoin: Thu thập từ CoinGecko API
## 3. Tiền xử lý dữ liệu

-   Loại bỏ ký tự đặc biệt, chuẩn hóa văn bản.\
-   Tính toán sentiment cho từng tweet:
    -   **TextBlob:** polarity (-1 → 1), subjectivity (0 → 1).\
    -   **VADER Sentiment:** compound, neg, neu, pos.\

## 4. Tạo nhãn mục tiêu

-   **Price Indicator:** dựa vào biến động giá (Close - Open).\
-   Tạo cột `target`:
    -   `0`: giá giảm\
    -   `1`: giá tăng

## 5. Xây dựng mô hình

- Combine sentiment features với price features
- Train/test split (80/20)
- Thử nghiệm các thuật toán ML khác nhau

## 6. Kết quả và kết luận



------------------------------------------------------------------------

### Tóm tắt


