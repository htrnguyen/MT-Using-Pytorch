#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script chính để chạy mô hình dịch máy Tiếng Việt - Tiếng Anh trên Kaggle
"""

import os
import argparse
import logging
import torch
import shutil
import pickle
import pandas as pd
from data_preprocessing import DataPreprocessor
from train import train_transformer, train_lstm, compare_models
from evaluate import evaluate_and_compare

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def prepare_data(config):
    """
    Chuẩn bị dữ liệu từ dataset vào thư mục làm việc
    
    Args:
        config (dict): Cấu hình
    """
    # Tạo thư mục
    os.makedirs(config['data_dir'], exist_ok=True)
    
    # Kiểm tra xem file dữ liệu đã tồn tại trong thư mục data chưa
    eng_vie_csv = os.path.join(config['data_dir'], 'eng_vie.csv')
    vie_txt = os.path.join(config['data_dir'], 'vie.txt')
    
    # Nếu chưa tồn tại, sao chép từ dataset
    if not os.path.exists(eng_vie_csv):
        # Đường dẫn file trong dataset
        dataset_eng_vie_csv = './data/eng_vie.csv'
        
        if os.path.exists(dataset_eng_vie_csv):
            shutil.copy(dataset_eng_vie_csv, eng_vie_csv)
            logger.info(f"Đã sao chép {dataset_eng_vie_csv} vào {eng_vie_csv}")
        else:
            logger.error(f"Không tìm thấy file {dataset_eng_vie_csv}")
            return False
    
    if not os.path.exists(vie_txt):
        # Đường dẫn file trong dataset
        dataset_vie_txt = './data/vie.txt'
        
        if os.path.exists(dataset_vie_txt):
            shutil.copy(dataset_vie_txt, vie_txt)
            logger.info(f"Đã sao chép {dataset_vie_txt} vào {vie_txt}")
        else:
            logger.error(f"Không tìm thấy file {dataset_vie_txt}")
            return False
    
    logger.info("Dữ liệu đã được chuẩn bị thành công")
    return True


def process_data(config):
    """
    Xử lý dữ liệu
    
    Args:
        config (dict): Cấu hình
    
    Returns:
        bool: Thành công hay không
    """
    # Kiểm tra xem file dữ liệu đã tồn tại chưa
    eng_vie_csv = os.path.join(config['data_dir'], 'eng_vie.csv')
    vie_txt = os.path.join(config['data_dir'], 'vie.txt')
    
    if not os.path.exists(eng_vie_csv) or not os.path.exists(vie_txt):
        logger.error(f"Không tìm thấy file dữ liệu. Vui lòng chạy lệnh với --prepare_data trước.")
        return False
    
    # Tạo đối tượng DataPreprocessor
    data_preprocessor = DataPreprocessor(config)
    
    # Xử lý dữ liệu
    try:
        train_data, val_data, test_data, en_sp, vi_sp = data_preprocessor.process(eng_vie_csv, vie_txt)
        logger.info("Đã xử lý dữ liệu thành công")
        return True
    except Exception as e:
        logger.error(f"Lỗi khi xử lý dữ liệu: {e}")
        return False


def train_models(config, model_type=None):
    """
    Huấn luyện mô hình
    
    Args:
        config (dict): Cấu hình
        model_type (str, optional): Loại mô hình ('transformer', 'lstm' hoặc None cho cả hai)
    
    Returns:
        bool: Thành công hay không
    """
    # Kiểm tra xem tokenizer đã tồn tại chưa
    tokenizers_path = os.path.join(config['model_dir'], 'tokenizers.pkl')
    
    if not os.path.exists(tokenizers_path):
        logger.error(f"Không tìm thấy tokenizer. Vui lòng xử lý dữ liệu trước.")
        return False
    
    # Huấn luyện mô hình
    try:
        transformer_history = None
        lstm_history = None
        
        if model_type is None or model_type == 'transformer':
            logger.info("Bắt đầu huấn luyện mô hình Transformer...")
            _, transformer_history = train_transformer(config)
        
        if model_type is None or model_type == 'lstm':
            logger.info("Bắt đầu huấn luyện mô hình LSTM...")
            _, lstm_history = train_lstm(config)
        
        # So sánh hai mô hình nếu cả hai đều được huấn luyện
        if transformer_history is not None and lstm_history is not None:
            compare_models(transformer_history, lstm_history, config)
        
        logger.info("Đã huấn luyện mô hình thành công")
        return True
    except Exception as e:
        logger.error(f"Lỗi khi huấn luyện mô hình: {e}")
        return False


def evaluate_models(config):
    """
    Đánh giá mô hình
    
    Args:
        config (dict): Cấu hình
    
    Returns:
        bool: Thành công hay không
    """
    # Kiểm tra xem mô hình đã tồn tại chưa
    transformer_path = os.path.join(config['model_dir'], 'transformer_best.pt')
    lstm_path = os.path.join(config['model_dir'], 'lstm_best.pt')
    
    if not os.path.exists(transformer_path) and not os.path.exists(lstm_path):
        logger.error(f"Không tìm thấy mô hình. Vui lòng huấn luyện mô hình trước.")
        return False
    
    # Đánh giá mô hình
    try:
        transformer_bleu, lstm_bleu = evaluate_and_compare(config)
        
        # Kết luận
        if transformer_bleu > lstm_bleu:
            logger.info("\nKết luận: Mô hình Transformer có hiệu suất tốt hơn!")
        elif lstm_bleu > transformer_bleu:
            logger.info("\nKết luận: Mô hình LSTM có hiệu suất tốt hơn!")
        else:
            logger.info("\nKết luận: Hai mô hình có hiệu suất tương đương nhau.")
        
        logger.info("Đã đánh giá mô hình thành công")
        return True
    except Exception as e:
        logger.error(f"Lỗi khi đánh giá mô hình: {e}")
        return False


def main():
    """
    Hàm chính
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='Mô hình dịch máy Tiếng Việt - Tiếng Anh')
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'prepare', 'process', 'train', 'evaluate'],
                        help='Chế độ chạy (all, prepare, process, train, evaluate)')
    parser.add_argument('--model_type', type=str, default=None, choices=['transformer', 'lstm'],
                        help='Loại mô hình (transformer, lstm hoặc None cho cả hai)')
    parser.add_argument('--data_dir', type=str, default='./data', help='Thư mục dữ liệu')
    parser.add_argument('--model_dir', type=str, default='./models', help='Thư mục mô hình')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Thiết bị (cuda, cpu)')
    parser.add_argument('--prepare_data', action='store_true', help='Chuẩn bị dữ liệu từ dataset')
    args = parser.parse_args()
    
    # Cấu hình
    config = {
        'data_dir': args.data_dir,
        'model_dir': args.model_dir,
        'src_lang': 'en',
        'tgt_lang': 'vi',
        'batch_size': 32,
        'max_len': 128,
        'num_workers': 2,
        'device': args.device,
        'mixed_precision': True,
        'gradient_accumulation_steps': 4,
        'min_sentence_length': 3,
        'max_sentence_length': 100,
        'length_ratio_threshold': 2.5,
        'vocab_size': 8000,
        'tokenizer_type': 'bpe',
        'val_test_size': 0.2,
        'random_seed': 42,
        
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
            'epochs': 20,
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
            'epochs': 20,
            'patience': 5
        }
    }
    
    # Tạo thư mục
    os.makedirs(config['data_dir'], exist_ok=True)
    os.makedirs(config['model_dir'], exist_ok=True)
    
    # Chuẩn bị dữ liệu nếu cần
    if args.prepare_data:
        logger.info("Chuẩn bị dữ liệu từ dataset...")
        if not prepare_data(config):
            return
    
    # Chạy theo chế độ
    if args.mode == 'all' or args.mode == 'prepare':
        logger.info("Chuẩn bị dữ liệu từ dataset...")
        if not prepare_data(config):
            return
    
    if args.mode == 'all' or args.mode == 'process':
        logger.info("Xử lý dữ liệu...")
        if not process_data(config):
            return
    
    if args.mode == 'all' or args.mode == 'train':
        logger.info("Huấn luyện mô hình...")
        if not train_models(config, args.model_type):
            return
    
    if args.mode == 'all' or args.mode == 'evaluate':
        logger.info("Đánh giá mô hình...")
        if not evaluate_models(config):
            return
    
    logger.info("Hoàn thành!")


if __name__ == "__main__":
    main()
