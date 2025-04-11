# Hướng dẫn chạy trên Kaggle

Để chạy mã nguồn này trên Kaggle, vui lòng làm theo các bước sau:

## 1. Chuẩn bị dữ liệu

Đảm bảo bạn đã tạo thư mục `data` trong thư mục làm việc của bạn và đã sao chép các file dữ liệu vào đó:
- `data/eng_vie.csv`: File CSV chứa cặp câu tiếng Anh - tiếng Việt
- `data/vie.txt`: File TSV chứa cặp câu tiếng Anh - tiếng Việt với metadata

Bạn có thể sử dụng lệnh sau để sao chép dữ liệu từ dataset vào thư mục làm việc:
```python
!mkdir -p ./data
!cp /kaggle/input/eng-viet/eng_vie.csv ./data/
!cp /kaggle/input/eng-viet/vie.txt ./data/
```

## 2. Tải mã nguồn lên Kaggle

Tải tất cả các file mã nguồn sau lên Kaggle:
- `data_preprocessing.py`
- `transformer_model.py`
- `lstm_model.py`
- `train.py`
- `evaluate.py`
- `kaggle_main.py` (đã được điều chỉnh đường dẫn cho Kaggle)

## 3. Chạy mã nguồn

Trong notebook Kaggle, chạy lệnh sau:

```python
!python kaggle_main.py --mode all --device cuda
```

Các tham số khác có thể điều chỉnh:
- `--mode`: Chọn chế độ chạy (`preprocess`, `train`, `evaluate`, hoặc `all`)
- `--model_type`: Chọn loại mô hình (`transformer`, `lstm`, hoặc `both`)
- `--batch_size`: Kích thước batch (mặc định: 32)
- `--epochs`: Số epoch (mặc định: 20)
- `--vocab_size`: Kích thước từ điển (mặc định: 8000)
- `--mixed_precision`: Sử dụng mixed precision để tăng tốc và tiết kiệm bộ nhớ
- `--gradient_accumulation_steps`: Số bước tích lũy gradient (mặc định: 4)

## 4. Cấu trúc thư mục trên Kaggle

Mã nguồn sẽ tự động tạo và sử dụng các thư mục sau:
- `./data`: Thư mục chứa dữ liệu đầu vào và dữ liệu đã xử lý
- `./models`: Thư mục chứa mô hình đã huấn luyện

## 5. Lưu ý quan trọng

- Đảm bảo GPU T4 được bật trong notebook Kaggle
- Nếu gặp vấn đề về bộ nhớ, hãy giảm `batch_size` và tăng `gradient_accumulation_steps`
- Quá trình huấn luyện có thể mất nhiều thời gian, hãy cân nhắc chỉ chạy một loại mô hình bằng cách sử dụng tham số `--model_type`

## 6. Ví dụ chạy từng bước

### Chỉ tiền xử lý dữ liệu:
```python
!python kaggle_main.py --mode preprocess
```

### Chỉ huấn luyện mô hình Transformer:
```python
!python kaggle_main.py --mode train --model_type transformer
```

### Chỉ huấn luyện mô hình LSTM:
```python
!python kaggle_main.py --mode train --model_type lstm
```

### Chỉ đánh giá mô hình:
```python
!python kaggle_main.py --mode evaluate
```

## 7. Kết quả

Sau khi chạy xong, kết quả sẽ được lưu trong thư mục `./models/`:
- Mô hình đã huấn luyện (`.pt`)
- Biểu đồ lịch sử huấn luyện (`.png`)
- Kết quả so sánh (`.csv`)
- Báo cáo đánh giá (`.csv`)

Bạn có thể tải xuống các file này từ tab "Output" trong notebook Kaggle.
