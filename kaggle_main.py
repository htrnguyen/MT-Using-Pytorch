#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script chính để chạy huấn luyện và đánh giá mô hình dịch máy Tiếng Việt - Tiếng Anh trên Kaggle
"""

import os
import argparse
import torch
import logging
import shutil

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def main():
    """
    Hàm chính để chạy huấn luyện và đánh giá mô hình
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='Huấn luyện và đánh giá mô hình dịch máy Tiếng Việt - Tiếng Anh')
    parser.add_argument('--mode', type=str, choices=['preprocess', 'train', 'evaluate', 'all'], default='all',
                        help='Chế độ chạy: preprocess, train, evaluate hoặc all')
    parser.add_argument('--data_dir', type=str, default='./data', help='Thư mục dữ liệu')
    parser.add_argument('--model_dir', type=str, default='./models', help='Thư mục mô hình')
    parser.add_argument('--src_lang', type=str, default='en', help='Ngôn ngữ nguồn')
    parser.add_argument('--tgt_lang', type=str, default='vi', help='Ngôn ngữ đích')
    parser.add_argument('--batch_size', type=int, default=32, help='Kích thước batch')
    parser.add_argument('--epochs', type=int, default=20, help='Số epoch')
    parser.add_argument('--model_type', type=str, choices=['transformer', 'lstm', 'both'], default='both',
                        help='Loại mô hình: transformer, lstm hoặc both')
    parser.add_argument('--vocab_size', type=int, default=8000, help='Kích thước từ điển')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Thiết bị (CPU/GPU)')
    parser.add_argument('--num_workers', type=int, default=2, help='Số worker cho DataLoader')
    parser.add_argument('--mixed_precision', action='store_true', help='Sử dụng mixed precision')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Số bước tích lũy gradient')
    parser.add_argument('--prepare_data', action='store_true', help='Chuẩn bị dữ liệu từ dataset')
    args = parser.parse_args()
    
    # Chuẩn bị dữ liệu từ dataset nếu cần
    if args.prepare_data:
        logger.info("Chuẩn bị dữ liệu từ dataset...")
        os.makedirs(args.data_dir, exist_ok=True)
        try:
            shutil.copy('/kaggle/input/eng-viet/eng_vie.csv', os.path.join(args.data_dir, 'eng_vie.csv'))
            shutil.copy('/kaggle/input/eng-viet/vie.txt', os.path.join(args.data_dir, 'vie.txt'))
            logger.info("Đã sao chép dữ liệu từ dataset vào thư mục làm việc")
        except Exception as e:
            logger.error(f"Lỗi khi sao chép dữ liệu: {e}")
            logger.info("Vui lòng đảm bảo dataset 'eng-viet' đã được thêm vào notebook")
    
    # Tạo thư mục
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Cấu hình
    config = {
        'data_dir': args.data_dir,
        'model_dir': args.model_dir,
        'src_lang': args.src_lang,
        'tgt_lang': args.tgt_lang,
        'batch_size': args.batch_size,
        'max_len': 128,
        'num_workers': args.num_workers,
        'device': args.device,
        'mixed_precision': args.mixed_precision,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        
        'transformer': {
            'd_model': 512,
            'nhead': 8,
            'num_encoder_layers': 6,
            'num_decoder_layers': 6,
            'd_ff': 2048,
            'dropout': 0.1,
            'pad_idx': 0,
            'label_smoothing': 0.1,
            'learning_rate': 0.0001,
            'lr_factor': 2.0,
            'warmup_steps': 4000,
            'clip_grad': 1.0,
            'epochs': args.epochs,
            'patience': 5
        },
        
        'lstm': {
            'embed_size': 256,
            'hidden_size': 512,
            'enc_num_layers': 2,
            'dec_num_layers': 2,
            'dropout': 0.3,
            'bidirectional': True,
            'pad_idx': 0,
            'learning_rate': 0.001,
            'clip_grad': 1.0,
            'epochs': args.epochs,
            'patience': 5
        }
    }
    
    # Tiền xử lý dữ liệu
    if args.mode in ['preprocess', 'all']:
        logger.info("Bắt đầu tiền xử lý dữ liệu...")
        from data_preprocessing import DataPreprocessor
        
        preprocessor_config = {
            'data_dir': args.data_dir,
            'model_dir': args.model_dir,
            'vocab_size': args.vocab_size,
            'tokenizer_type': 'bpe',
            'min_sentence_length': 3,
            'max_sentence_length': 100,
            'length_ratio_threshold': 2.5,
            'val_test_size': 0.2,
            'random_seed': 42,
            'src_lang': args.src_lang,
            'tgt_lang': args.tgt_lang,
            'batch_size': args.batch_size,
            'max_len': 128,
            'num_workers': args.num_workers
        }
        
        preprocessor = DataPreprocessor(preprocessor_config)
        
        # Đường dẫn file dữ liệu
        file1_path = os.path.join(args.data_dir, 'eng_vie.csv')
        file2_path = os.path.join(args.data_dir, 'vie.txt')
        
        # Kiểm tra xem file dữ liệu có tồn tại không
        if not os.path.exists(file1_path) or not os.path.exists(file2_path):
            logger.error(f"Không tìm thấy file dữ liệu. Vui lòng đảm bảo các file sau tồn tại:")
            logger.error(f"- {file1_path}")
            logger.error(f"- {file2_path}")
            logger.info("Bạn có thể chạy lại với tham số --prepare_data để sao chép dữ liệu từ dataset")
            return
        
        # Tiền xử lý dữ liệu
        train_data, val_data, test_data, en_sp, vi_sp = preprocessor.process(file1_path, file2_path)
        logger.info("Hoàn thành tiền xử lý dữ liệu!")
    
    # Huấn luyện mô hình
    if args.mode in ['train', 'all']:
        logger.info("Bắt đầu huấn luyện mô hình...")
        from train import train_transformer, train_lstm
        
        # Huấn luyện mô hình Transformer
        if args.model_type in ['transformer', 'both']:
            logger.info("Huấn luyện mô hình Transformer...")
            transformer_model, transformer_history = train_transformer(config)
        
        # Huấn luyện mô hình LSTM
        if args.model_type in ['lstm', 'both']:
            logger.info("Huấn luyện mô hình LSTM...")
            lstm_model, lstm_history = train_lstm(config)
        
        # So sánh mô hình
        if args.model_type == 'both':
            from train import compare_models
            compare_models(transformer_history, lstm_history, config)
        
        logger.info("Hoàn thành huấn luyện mô hình!")
    
    # Đánh giá mô hình
    if args.mode in ['evaluate', 'all']:
        logger.info("Bắt đầu đánh giá mô hình...")
        from evaluate import main as evaluate_main
        
        # Cấu hình đánh giá
        eval_config = {
            'data_dir': args.data_dir,
            'model_dir': args.model_dir,
            'num_samples': 100,
            'device': args.device,
            'src_lang': args.src_lang,
            'tgt_lang': args.tgt_lang,
            
            'transformer': config['transformer'],
            'lstm': config['lstm']
        }
        
        # Đánh giá mô hình
        best_model, transformer_bleu, lstm_bleu = evaluate_main(eval_config)
        logger.info(f"Mô hình tốt nhất: {best_model}")
        logger.info(f"BLEU score của mô hình Transformer: {transformer_bleu:.2f}")
        logger.info(f"BLEU score của mô hình LSTM: {lstm_bleu:.2f}")
        
        logger.info("Hoàn thành đánh giá mô hình!")
    
    logger.info("Hoàn thành tất cả các bước!")


if __name__ == "__main__":
    main()
