# Mô hình dịch máy Tiếng Việt - Tiếng Anh sử dụng Transformer và LSTM

Dự án này xây dựng và so sánh hai mô hình dịch máy từ Tiếng Anh sang Tiếng Việt sử dụng kiến trúc Transformer và LSTM. Mã nguồn được tối ưu hóa để chạy trên GPU T4 của Kaggle, với các kỹ thuật chống overfitting và quản lý tài nguyên GPU/RAM hiệu quả.

## Tính năng chính

- **Tiền xử lý dữ liệu nâng cao**:
  - Chuẩn hóa Unicode cho tiếng Việt (NFC format)
  - Xử lý đúng từ ghép tiếng Việt (như "Hà Nội") thông qua user_defined_symbols
  - Lọc câu quá dài/ngắn và tỷ lệ độ dài bất thường

- **Mô hình Transformer cải tiến**:
  - Kiến trúc Transformer với nhiều tính năng tối ưu
  - Label smoothing để tránh overfitting
  - Gradient checkpointing để tiết kiệm bộ nhớ
  - NoamOpt scheduler cho learning rate

- **Mô hình LSTM với Attention**:
  - LSTM hai chiều cho encoder
  - Cơ chế Attention cho decoder
  - Tối ưu hóa để tránh overfitting

- **Huấn luyện tối ưu**:
  - Mixed precision training
  - Gradient accumulation
  - Early stopping và model checkpointing
  - Trực quan hóa quá trình huấn luyện

- **Đánh giá và so sánh**:
  - Đánh giá bằng BLEU score
  - So sánh hiệu suất giữa Transformer và LSTM
  - Phân tích lỗi và đề xuất cải tiến

## Cấu trúc dự án

```
translation_project/
├── data_preprocessing.py  # Tiền xử lý dữ liệu
├── transformer_model.py   # Mô hình Transformer
├── lstm_model.py          # Mô hình LSTM
├── train.py               # Huấn luyện mô hình
├── evaluate.py            # Đánh giá và so sánh mô hình
├── main.py                # Script chính để chạy toàn bộ quy trình
├── data/                  # Thư mục chứa dữ liệu đã xử lý
└── models/                # Thư mục chứa mô hình đã huấn luyện
```

## Yêu cầu

- Python 3.6+
- PyTorch 1.7+
- sentencepiece
- sacrebleu
- matplotlib
- seaborn
- pandas
- numpy
- tqdm
- tabulate

## Cài đặt

```bash
pip install torch sentencepiece sacrebleu matplotlib seaborn pandas numpy tqdm tabulate
```

## Hướng dẫn sử dụng

### Chạy toàn bộ quy trình

```bash
python main.py --mode all
```

### Chỉ tiền xử lý dữ liệu

```bash
python main.py --mode preprocess
```

### Chỉ huấn luyện mô hình

```bash
python main.py --mode train --model_type both  # huấn luyện cả hai mô hình
python main.py --mode train --model_type transformer  # chỉ huấn luyện Transformer
python main.py --mode train --model_type lstm  # chỉ huấn luyện LSTM
```

### Chỉ đánh giá mô hình

```bash
python main.py --mode evaluate
```

### Các tham số khác

```
--data_dir: Thư mục dữ liệu (mặc định: ./data)
--model_dir: Thư mục mô hình (mặc định: ./models)
--src_lang: Ngôn ngữ nguồn (mặc định: en)
--tgt_lang: Ngôn ngữ đích (mặc định: vi)
--batch_size: Kích thước batch (mặc định: 32)
--epochs: Số epoch (mặc định: 20)
--vocab_size: Kích thước từ điển (mặc định: 8000)
--device: Thiết bị (mặc định: cuda nếu có, ngược lại cpu)
--num_workers: Số worker cho DataLoader (mặc định: 2)
--mixed_precision: Sử dụng mixed precision (mặc định: False)
--gradient_accumulation_steps: Số bước tích lũy gradient (mặc định: 4)
```

## Sử dụng trên Kaggle

1. Tải mã nguồn lên Kaggle
2. Đảm bảo dữ liệu `eng_vie.csv` và `vie.txt` được đặt đúng vị trí
3. Chạy với GPU T4:

```python
!python main.py --mode all --device cuda
```

## Kết quả

Sau khi chạy xong, kết quả sẽ được lưu trong thư mục `models/`:

- `transformer_best.pt`: Mô hình Transformer tốt nhất
- `lstm_best.pt`: Mô hình LSTM tốt nhất
- `transformer_history.png`: Biểu đồ lịch sử huấn luyện Transformer
- `lstm_history.png`: Biểu đồ lịch sử huấn luyện LSTM
- `model_comparison.png`: Biểu đồ so sánh hai mô hình
- `model_comparison.csv`: Bảng so sánh chi tiết
- `translation_comparison.csv`: So sánh kết quả dịch
- `custom_translations.csv`: Kết quả dịch các câu tùy chỉnh

## Tùy chỉnh mô hình

Để tùy chỉnh cấu hình mô hình, bạn có thể chỉnh sửa các tham số trong file `main.py`:

```python
config = {
    'transformer': {
        'd_model': 512,  # Kích thước embedding
        'nhead': 8,      # Số lượng head trong Multi-Head Attention
        # ...
    },
    'lstm': {
        'embed_size': 256,  # Kích thước embedding
        'hidden_size': 512, # Kích thước hidden state
        # ...
    }
}
```

## Xử lý vấn đề về bộ nhớ GPU

Nếu gặp vấn đề về bộ nhớ GPU, bạn có thể thử các giải pháp sau:

1. Giảm kích thước batch: `--batch_size 16`
2. Tăng số bước tích lũy gradient: `--gradient_accumulation_steps 8`
3. Sử dụng mixed precision: `--mixed_precision`
4. Giảm kích thước mô hình bằng cách chỉnh sửa cấu hình trong `main.py`

## Tham khảo

Dự án này được phát triển dựa trên tham khảo từ:
- [Language Translation using PyTorch Transformer](https://debuggercafe.com/language-translation-using-pytorch-transformer/)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [PyTorch Seq2Seq Tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

## Tác giả

Dự án được phát triển bởi Manus AI theo yêu cầu của người dùng.
