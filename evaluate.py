#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Đánh giá mô hình dịch máy Tiếng Việt - Tiếng Anh
"""

import os
import logging
import torch
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sacrebleu
from tqdm import tqdm

# Import các module đã tạo
from data_preprocessing import create_dataloaders
from transformer_model import translate as transformer_translate
from lstm_model import LSTMSeq2Seq

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def load_model(model_path, model_type, config, src_vocab_size, tgt_vocab_size):
    """
    Tải mô hình từ file
    
    Args:
        model_path (str): Đường dẫn đến file mô hình
        model_type (str): Loại mô hình ('transformer' hoặc 'lstm')
        config (dict): Cấu hình mô hình
        src_vocab_size (int): Kích thước từ điển nguồn
        tgt_vocab_size (int): Kích thước từ điển đích
    
    Returns:
        model: Mô hình đã tải
    """
    # Tạo mô hình
    if model_type == 'transformer':
        from transformer_model import create_transformer_model
        model = create_transformer_model(config['transformer'], src_vocab_size, tgt_vocab_size)
    else:
        from lstm_model import create_lstm_model
        model = create_lstm_model(config['lstm'], src_vocab_size, tgt_vocab_size)
    
    # Tải trọng số
    checkpoint = torch.load(model_path, map_location=torch.device(config['device']))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Chuyển sang thiết bị
    model = model.to(config['device'])
    
    # Chuyển sang chế độ đánh giá
    model.eval()
    
    return model


def calculate_bleu(references, hypotheses):
    """
    Tính điểm BLEU
    
    Args:
        references (list): Danh sách các câu tham chiếu
        hypotheses (list): Danh sách các câu dự đoán
    
    Returns:
        float: Điểm BLEU
    """
    # Tính điểm BLEU
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    
    return bleu.score


def evaluate_model(model, test_loader, tokenizers, config, model_type='transformer', num_samples=100):
    """
    Đánh giá mô hình trên tập test
    
    Args:
        model: Mô hình cần đánh giá
        test_loader: DataLoader cho tập test
        tokenizers (dict): Các tokenizer
        config (dict): Cấu hình
        model_type (str): Loại mô hình ('transformer' hoặc 'lstm')
        num_samples (int): Số lượng mẫu để đánh giá
    
    Returns:
        tuple: (bleu_score, examples) - Điểm BLEU và các ví dụ
    """
    # Xác định tokenizer
    src_sp = tokenizers['en_sp'] if config['src_lang'] == 'en' else tokenizers['vi_sp']
    tgt_sp = tokenizers['vi_sp'] if config['tgt_lang'] == 'vi' else tokenizers['en_sp']
    
    # Danh sách lưu kết quả
    references = []
    hypotheses = []
    examples = []
    
    # Đánh giá từng batch
    for i, (src, tgt, src_lengths, tgt_lengths) in enumerate(tqdm(test_loader, desc="Evaluating")):
        # Chỉ đánh giá một số lượng mẫu nhất định
        if i * test_loader.batch_size >= num_samples:
            break
        
        # Chuyển dữ liệu sang thiết bị
        src = src.to(config['device'])
        tgt = tgt.to(config['device'])
        src_lengths = src_lengths.to(config['device'])
        
        # Dịch câu
        with torch.no_grad():
            if model_type == 'transformer':
                # Dịch từng câu trong batch
                batch_translations = []
                for j in range(src.size(0)):
                    translation = transformer_translate(
                        model, 
                        src_sp.decode(src[j].cpu().numpy().tolist()), 
                        src_sp, 
                        tgt_sp, 
                        config['device']
                    )
                    batch_translations.append(translation)
            else:
                # LSTM model
                batch_translations, _ = model.translate(src, src_lengths, tgt_sp, device=config['device'])
        
        # Lấy câu tham chiếu
        batch_references = []
        for j in range(tgt.size(0)):
            # Bỏ qua token BOS, EOS và PAD
            tgt_tokens = tgt[j].cpu().numpy().tolist()
            tgt_tokens = [t for t in tgt_tokens if t > 3]  # Bỏ qua PAD, UNK, BOS, EOS
            reference = tgt_sp.decode(tgt_tokens)
            batch_references.append(reference)
        
        # Thêm vào danh sách kết quả
        references.extend(batch_references)
        hypotheses.extend(batch_translations)
        
        # Lưu một số ví dụ
        for j in range(min(3, len(batch_translations))):
            if len(examples) < 10:  # Chỉ lưu 10 ví dụ
                src_text = src_sp.decode([t for t in src[j].cpu().numpy().tolist() if t > 3])
                examples.append({
                    'source': src_text,
                    'reference': batch_references[j],
                    'translation': batch_translations[j]
                })
    
    # Tính điểm BLEU
    bleu_score = calculate_bleu(references, hypotheses)
    
    return bleu_score, examples


def compare_translations(transformer_examples, lstm_examples, config):
    """
    So sánh kết quả dịch của hai mô hình
    
    Args:
        transformer_examples (list): Các ví dụ dịch của Transformer
        lstm_examples (list): Các ví dụ dịch của LSTM
        config (dict): Cấu hình
    """
    # Tạo DataFrame so sánh
    comparison = []
    
    for i in range(min(len(transformer_examples), len(lstm_examples))):
        comparison.append({
            'Source': transformer_examples[i]['source'],
            'Reference': transformer_examples[i]['reference'],
            'Transformer': transformer_examples[i]['translation'],
            'LSTM': lstm_examples[i]['translation']
        })
    
    comparison_df = pd.DataFrame(comparison)
    
    # Lưu DataFrame
    comparison_df.to_csv(os.path.join(config['model_dir'], "translation_comparison.csv"), index=False)
    
    # In một số ví dụ
    logger.info("\nSo sánh kết quả dịch của hai mô hình:")
    for i, row in comparison_df.head(3).iterrows():
        logger.info(f"\nVí dụ {i+1}:")
        logger.info(f"Source: {row['Source']}")
        logger.info(f"Reference: {row['Reference']}")
        logger.info(f"Transformer: {row['Transformer']}")
        logger.info(f"LSTM: {row['LSTM']}")


def evaluate_and_compare(config):
    """
    Đánh giá và so sánh hai mô hình
    
    Args:
        config (dict): Cấu hình
    
    Returns:
        tuple: (transformer_bleu, lstm_bleu) - Điểm BLEU của hai mô hình
    """
    # Tải tokenizer
    with open(os.path.join(config['model_dir'], 'tokenizers.pkl'), 'rb') as f:
        tokenizers = pickle.load(f)
    
    en_sp = tokenizers['en_sp']
    vi_sp = tokenizers['vi_sp']
    
    # Tải dữ liệu
    test_data = pd.read_csv(os.path.join(config['data_dir'], 'test_data.csv'))
    
    # Tạo DataLoader
    _, _, test_loader = create_dataloaders(
        None, None, test_data, en_sp, vi_sp, config
    )
    
    # Xác định kích thước từ điển
    src_vocab_size = en_sp.get_piece_size() if config['src_lang'] == 'en' else vi_sp.get_piece_size()
    tgt_vocab_size = vi_sp.get_piece_size() if config['tgt_lang'] == 'vi' else en_sp.get_piece_size()
    
    # Tải mô hình Transformer
    transformer_path = os.path.join(config['model_dir'], "transformer_best.pt")
    if os.path.exists(transformer_path):
        transformer_model = load_model(transformer_path, 'transformer', config, src_vocab_size, tgt_vocab_size)
        
        # Đánh giá mô hình Transformer
        logger.info("Đánh giá mô hình Transformer...")
        transformer_bleu, transformer_examples = evaluate_model(
            transformer_model, test_loader, tokenizers, config, model_type='transformer'
        )
        logger.info(f"Transformer BLEU: {transformer_bleu:.2f}")
    else:
        logger.warning(f"Không tìm thấy mô hình Transformer tại {transformer_path}")
        transformer_bleu = 0
        transformer_examples = []
    
    # Tải mô hình LSTM
    lstm_path = os.path.join(config['model_dir'], "lstm_best.pt")
    if os.path.exists(lstm_path):
        lstm_model = load_model(lstm_path, 'lstm', config, src_vocab_size, tgt_vocab_size)
        
        # Đánh giá mô hình LSTM
        logger.info("Đánh giá mô hình LSTM...")
        lstm_bleu, lstm_examples = evaluate_model(
            lstm_model, test_loader, tokenizers, config, model_type='lstm'
        )
        logger.info(f"LSTM BLEU: {lstm_bleu:.2f}")
    else:
        logger.warning(f"Không tìm thấy mô hình LSTM tại {lstm_path}")
        lstm_bleu = 0
        lstm_examples = []
    
    # So sánh kết quả dịch
    if transformer_examples and lstm_examples:
        compare_translations(transformer_examples, lstm_examples, config)
    
    # Vẽ biểu đồ so sánh BLEU
    if transformer_bleu > 0 or lstm_bleu > 0:
        plt.figure(figsize=(8, 6))
        models = ['Transformer', 'LSTM']
        bleu_scores = [transformer_bleu, lstm_bleu]
        
        sns.barplot(x=models, y=bleu_scores)
        plt.title('BLEU Score Comparison')
        plt.ylabel('BLEU Score')
        plt.ylim(0, max(bleu_scores) * 1.2)
        
        # Thêm giá trị lên đầu cột
        for i, score in enumerate(bleu_scores):
            plt.text(i, score + 1, f"{score:.2f}", ha='center')
        
        # Lưu biểu đồ
        plt.tight_layout()
        plt.savefig(os.path.join(config['model_dir'], "bleu_comparison.png"))
        plt.close()
    
    return transformer_bleu, lstm_bleu


if __name__ == "__main__":
    # Cấu hình
    config = {
        'data_dir': './data',
        'model_dir': './models',
        'src_lang': 'en',
        'tgt_lang': 'vi',
        'batch_size': 32,
        'max_len': 128,
        'num_workers': 2,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        'transformer': {
            'd_model': 512,
            'nhead': 8,
            'num_encoder_layers': 6,
            'num_decoder_layers': 6,
            'd_ff': 2048,
            'dropout': 0.1,
            'pad_idx': 0
        },
        
        'lstm': {
            'embed_size': 256,
            'hidden_size': 512,
            'enc_num_layers': 2,
            'dec_num_layers': 2,
            'dropout': 0.3,
            'bidirectional': True,
            'pad_idx': 0
        }
    }
    
    # Đánh giá và so sánh hai mô hình
    transformer_bleu, lstm_bleu = evaluate_and_compare(config)
    
    # Kết luận
    if transformer_bleu > lstm_bleu:
        logger.info("\nKết luận: Mô hình Transformer có hiệu suất tốt hơn!")
    elif lstm_bleu > transformer_bleu:
        logger.info("\nKết luận: Mô hình LSTM có hiệu suất tốt hơn!")
    else:
        logger.info("\nKết luận: Hai mô hình có hiệu suất tương đương nhau.")
