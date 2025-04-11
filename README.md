# Mô hình dịch máy Tiếng Việt - Tiếng Anh

Dự án này xây dựng và so sánh hai mô hình dịch máy Tiếng Việt - Tiếng Anh sử dụng kiến trúc Transformer và LSTM. Mã nguồn được tối ưu hóa để chạy trên Kaggle với GPU T4.

## Tính năng chính

- **Tiền xử lý dữ liệu nâng cao**:
  - Chuẩn hóa Unicode cho tiếng Việt (NFC format)
  - Xử lý đúng từ ghép tiếng Việt (như "Hà Nội") thông qua user_defined_symbols
  - Lọc câu quá dài/ngắn và tỷ lệ độ dài bất thường

- **Mô hình Transformer**:
  - Kiến trúc Transformer đầy đủ với Multi-Head Attention
  - Label smoothing để tránh overfit
  - Gradient checkpointing để tiết kiệm bộ nhớ GPU

- **Mô hình LSTM**:
  - LSTM hai chiều với cơ chế Attention
  - Tối ưu hóa để tránh overfit và quản lý tài nguyên GPU

- **Huấn luyện hiệu quả**:
  - Mixed precision training
  - Gradient accumulation
  - Early stopping và model checkpointing

- **Đánh giá và so sánh**:
  - Đánh giá bằng BLEU score
  - So sánh hiệu suất giữa Transformer và LSTM

## Cấu trúc dự án

```
.
├── data/                  # Thư mục chứa dữ liệu
│   ├── eng_vie.csv        # File dữ liệu gốc
│   └── vie.txt            # File dữ liệu gốc
├── models/                # Thư mục chứa mô hình
├── data_preprocessing.py  # Tiền xử lý dữ liệu
├── transformer_model.py   # Mô hình Transformer
├── lstm_model.py          # Mô hình LSTM
├── train.py               # Huấn luyện mô hình
├── evaluate.py            # Đánh giá và so sánh mô hình
└── main.py                # Script chính để chạy
```

## Yêu cầu

- Python 3.6+
- PyTorch 1.8+
- sentencepiece
- sacrebleu
- pandas
- numpy
- matplotlib
- seaborn
- tqdm

## Cách sử dụng

### Chuẩn bị dữ liệu

```bash
python main.py --prepare_data
```

### Xử lý dữ liệu

```bash
python main.py --mode process
```

### Huấn luyện mô hình

```bash
# Huấn luyện cả hai mô hình
python main.py --mode train --device cuda

# Chỉ huấn luyện Transformer
python main.py --mode train --model_type transformer --device cuda

# Chỉ huấn luyện LSTM
python main.py --mode train --model_type lstm --device cuda
```

### Đánh giá mô hình

```bash
python main.py --mode evaluate --device cuda
```

### Chạy toàn bộ quy trình

```bash
python main.py --mode all --device cuda
```

## Tối ưu hóa cho Kaggle

Mã nguồn đã được tối ưu hóa để chạy trên Kaggle với GPU T4:

1. **Quản lý bộ nhớ GPU**:
   - Mixed precision training
   - Gradient accumulation
   - Gradient checkpointing

2. **Xử lý dữ liệu hiệu quả**:
   - Tokenization với sentencepiece
   - Xử lý đặc biệt cho tiếng Việt

3. **Tránh overfit**:
   - Label smoothing
   - Dropout
   - Early stopping

## Kết quả

Sau khi huấn luyện, kết quả so sánh giữa hai mô hình sẽ được lưu trong thư mục `models/`:
- `model_comparison.png`: Biểu đồ so sánh loss
- `bleu_comparison.png`: Biểu đồ so sánh điểm BLEU
- `translation_comparison.csv`: So sánh kết quả dịch của hai mô hình

## Tham khảo

- [Language Translation Using PyTorch Transformer](https://debuggercafe.com/language-translation-using-pytorch-transformer/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)
