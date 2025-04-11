# Kế hoạch xây dựng mô hình dịch máy Transformer và LSTM

## Phân tích dữ liệu và mã nguồn
- [x] Phân tích mã nguồn transformer.ipynb
- [x] Kiểm tra cấu trúc dữ liệu vie.txt
- [x] Kiểm tra cấu trúc dữ liệu eng_vie.csv
- [x] Tạo thư mục dự án

## Tiền xử lý dữ liệu
- [x] Chuẩn hóa dữ liệu tiếng Việt (chữ thường, dấu câu)
- [x] Cải thiện tokenizer để xử lý đúng từ ghép tiếng Việt
- [x] Xử lý dữ liệu trùng lặp và nhiễu
- [x] Chia tập dữ liệu thành train/validation/test
- [x] Tối ưu hóa quá trình xử lý dữ liệu để tiết kiệm bộ nhớ

## Triển khai mô hình Transformer
- [x] Cải thiện kiến trúc Transformer từ mã nguồn hiện có
- [x] Thêm cơ chế regularization để tránh overfitting
- [x] Tối ưu hóa hyperparameters
- [x] Thêm cơ chế quản lý bộ nhớ GPU

## Triển khai mô hình LSTM
- [x] Xây dựng kiến trúc mô hình LSTM Encoder-Decoder
- [x] Thêm cơ chế Attention cho LSTM
- [x] Tối ưu hóa hyperparameters
- [x] Thêm cơ chế quản lý bộ nhớ GPU

## Huấn luyện và tối ưu hóa mô hình
- [x] Triển khai huấn luyện với gradient accumulation
- [x] Thêm learning rate scheduler
- [x] Triển khai early stopping và model checkpointing
- [x] Theo dõi và trực quan hóa quá trình huấn luyện

## Đánh giá và so sánh mô hình
- [x] Đánh giá mô hình bằng BLEU score
- [x] So sánh hiệu suất Transformer vs LSTM
- [x] Kiểm tra chất lượng dịch trên các câu mẫu
- [x] Phân tích lỗi và đề xuất cải tiến

## Tạo sản phẩm cuối cùng
- [x] Tổng hợp mã nguồn thành file Jupyter Notebook hoàn chỉnh
- [x] Tổ chức mã nguồn thành các file Python có cấu trúc rõ ràng
- [x] Viết hướng dẫn sử dụng
- [x] Tạo script để chạy huấn luyện và đánh giá mô hình
