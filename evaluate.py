#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Đánh giá và so sánh mô hình dịch máy Tiếng Việt - Tiếng Anh
"""

import os
import torch
import logging
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sacrebleu
from tqdm import tqdm
import argparse
import numpy as np
from tabulate import tabulate

# Import các module đã tạo
from transformer_model import TransformerModel, translate as transformer_translate
from lstm_model import LSTMSeq2Seq

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def load_model(model_path, model_type, config, device):
    """
    Tải mô hình đã huấn luyện
    
    Args:
        model_path (str): Đường dẫn đến file mô hình
        model_type (str): Loại mô hình ('transformer' hoặc 'lstm')
        config (dict): Cấu hình mô hình
        device (str): Thiết bị (CPU/GPU)
        
    Returns:
        model: Mô hình đã tải
    """
    # Tải tokenizer
    with open(os.path.join(config['model_dir'], 'tokenizers.pkl'), 'rb') as f:
        tokenizers = pickle.load(f)
    
    en_sp = tokenizers['en_sp']
    vi_sp = tokenizers['vi_sp']
    
    # Xác định kích thước từ điển
    src_vocab_size = en_sp.get_piece_size() if config['src_lang'] == 'en' else vi_sp.get_piece_size()
    tgt_vocab_size = vi_sp.get_piece_size() if config['tgt_lang'] == 'vi' else en_sp.get_piece_size()
    
    # Tạo mô hình
    if model_type == 'transformer':
        from transformer_model import create_transformer_model
        model = create_transformer_model(config['transformer'], src_vocab_size, tgt_vocab_size)
    else:
        from lstm_model import create_lstm_model
        model = create_lstm_model(config['lstm'], src_vocab_size, tgt_vocab_size)
    
    # Tải trọng số
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"Đã tải mô hình {model_type} từ {model_path}")
    
    return model, en_sp, vi_sp


def translate_sentence(sentence, model, src_sp, tgt_sp, model_type, device, max_len=100):
    """
    Dịch một câu
    
    Args:
        sentence (str): Câu cần dịch
        model: Mô hình dịch
        src_sp: Tokenizer nguồn
        tgt_sp: Tokenizer đích
        model_type (str): Loại mô hình ('transformer' hoặc 'lstm')
        device (str): Thiết bị (CPU/GPU)
        max_len (int): Độ dài tối đa của câu dịch
        
    Returns:
        str: Câu đã dịch
    """
    model.eval()
    
    with torch.no_grad():
        if model_type == 'transformer':
            # Tokenize câu nguồn
            src_tokens = [src_sp.bos_id()] + src_sp.encode(sentence, out_type=int) + [src_sp.eos_id()]
            src_tensor = torch.tensor(src_tokens).unsqueeze(0).to(device)
            src_mask = (src_tensor != 0).unsqueeze(1).unsqueeze(2).to(device)
            
            # Encoder
            enc_output = model.encoder(src_tensor, src_mask)
            
            # Bắt đầu với token BOS
            tgt_tokens = [tgt_sp.bos_id()]
            tgt_tensor = torch.tensor([tgt_tokens]).to(device)
            
            # Dịch từng token
            for _ in range(max_len):
                tgt_mask = model.make_tgt_mask(tgt_tensor)
                output = model.decoder(tgt_tensor, enc_output, tgt_mask, src_mask)
                output = model.generator(output)
                
                # Lấy token có xác suất cao nhất
                pred_token = output.argmax(2)[:, -1].item()
                tgt_tokens.append(pred_token)
                tgt_tensor = torch.tensor([tgt_tokens]).to(device)
                
                # Dừng nếu gặp token EOS
                if pred_token == tgt_sp.eos_id():
                    break
            
            # Chuyển từ tokens sang câu
            pred_tokens = tgt_tokens[1:]  # Bỏ token BOS
            if pred_tokens[-1] == tgt_sp.eos_id():
                pred_tokens = pred_tokens[:-1]  # Bỏ token EOS
            
            return tgt_sp.decode(pred_tokens)
        
        elif model_type == 'lstm':
            # Tokenize câu nguồn
            src_tokens = [src_sp.bos_id()] + src_sp.encode(sentence, out_type=int) + [src_sp.eos_id()]
            src_tensor = torch.tensor(src_tokens).unsqueeze(0).to(device)
            src_lengths = torch.tensor([len(src_tokens)]).to(device)
            
            # Dịch câu
            translations, _ = model.translate(src_tensor, src_lengths, tgt_sp, max_len, device)
            
            return translations[0]


def evaluate_on_test_set(model, test_data, src_sp, tgt_sp, model_type, device, max_len=100, num_samples=100):
    """
    Đánh giá mô hình trên tập test
    
    Args:
        model: Mô hình dịch
        test_data (pd.DataFrame): Dữ liệu test
        src_sp: Tokenizer nguồn
        tgt_sp: Tokenizer đích
        model_type (str): Loại mô hình ('transformer' hoặc 'lstm')
        device (str): Thiết bị (CPU/GPU)
        max_len (int): Độ dài tối đa của câu dịch
        num_samples (int): Số lượng mẫu để đánh giá
        
    Returns:
        tuple: (bleu_score, hypotheses, references, src_sentences)
    """
    model.eval()
    
    # Lấy mẫu ngẫu nhiên từ tập test
    if len(test_data) > num_samples:
        test_samples = test_data.sample(num_samples, random_state=42)
    else:
        test_samples = test_data
    
    # Xác định cột dữ liệu
    src_col = 'en_normalized' if model_type == 'transformer' else 'en_normalized'
    tgt_col = 'vi_normalized' if model_type == 'transformer' else 'vi_normalized'
    
    hypotheses = []
    references = []
    src_sentences = []
    
    for _, row in tqdm(test_samples.iterrows(), total=len(test_samples), desc=f"Đánh giá mô hình {model_type}"):
        src_sentence = row[src_col]
        tgt_sentence = row[tgt_col]
        
        # Dịch câu
        translation = translate_sentence(
            src_sentence, model, src_sp, tgt_sp, model_type, device, max_len
        )
        
        hypotheses.append(translation)
        references.append([tgt_sentence])
        src_sentences.append(src_sentence)
    
    # Tính BLEU score
    bleu = sacrebleu.corpus_bleu(hypotheses, references)
    
    return bleu.score, hypotheses, references, src_sentences


def compare_translations(src_sentences, transformer_translations, lstm_translations, references, output_path=None):
    """
    So sánh kết quả dịch của hai mô hình
    
    Args:
        src_sentences (list): Danh sách câu nguồn
        transformer_translations (list): Danh sách câu dịch bởi Transformer
        lstm_translations (list): Danh sách câu dịch bởi LSTM
        references (list): Danh sách câu tham chiếu
        output_path (str, optional): Đường dẫn để lưu kết quả
    """
    # Tạo DataFrame
    comparison_df = pd.DataFrame({
        'Source': src_sentences,
        'Reference': [ref[0] for ref in references],
        'Transformer': transformer_translations,
        'LSTM': lstm_translations
    })
    
    # Tính BLEU score cho từng câu
    transformer_bleu = []
    lstm_bleu = []
    
    for i in range(len(src_sentences)):
        transformer_bleu.append(sacrebleu.sentence_bleu(transformer_translations[i], [references[i][0]]).score)
        lstm_bleu.append(sacrebleu.sentence_bleu(lstm_translations[i], [references[i][0]]).score)
    
    comparison_df['Transformer BLEU'] = transformer_bleu
    comparison_df['LSTM BLEU'] = lstm_bleu
    comparison_df['Better Model'] = comparison_df.apply(
        lambda row: 'Transformer' if row['Transformer BLEU'] > row['LSTM BLEU'] else 
                   ('LSTM' if row['LSTM BLEU'] > row['Transformer BLEU'] else 'Equal'),
        axis=1
    )
    
    # Lưu kết quả
    if output_path:
        comparison_df.to_csv(output_path, index=False)
        logger.info(f"Đã lưu kết quả so sánh tại {output_path}")
    
    # Thống kê
    model_counts = comparison_df['Better Model'].value_counts()
    logger.info("\nThống kê mô hình tốt hơn:")
    for model, count in model_counts.items():
        logger.info(f"{model}: {count} câu ({count/len(comparison_df)*100:.2f}%)")
    
    return comparison_df


def visualize_comparison(comparison_df, output_path=None):
    """
    Trực quan hóa kết quả so sánh
    
    Args:
        comparison_df (pd.DataFrame): DataFrame chứa kết quả so sánh
        output_path (str, optional): Đường dẫn để lưu biểu đồ
    """
    # Tạo figure với 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot phân phối BLEU score
    sns.histplot(comparison_df['Transformer BLEU'], ax=ax1, label='Transformer', color='blue', alpha=0.6, kde=True)
    sns.histplot(comparison_df['LSTM BLEU'], ax=ax1, label='LSTM', color='orange', alpha=0.6, kde=True)
    ax1.set_xlabel('BLEU Score')
    ax1.set_ylabel('Số lượng câu')
    ax1.set_title('Phân phối BLEU Score')
    ax1.legend()
    
    # Plot tỷ lệ mô hình tốt hơn
    model_counts = comparison_df['Better Model'].value_counts()
    ax2.pie(model_counts, labels=model_counts.index, autopct='%1.1f%%', colors=['blue', 'orange', 'green'])
    ax2.set_title('Tỷ lệ mô hình tốt hơn')
    
    # Điều chỉnh layout
    plt.tight_layout()
    
    # Lưu biểu đồ
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Đã lưu biểu đồ so sánh tại {output_path}")
    else:
        plt.show()


def evaluate_on_custom_sentences(sentences, transformer_model, lstm_model, src_sp, tgt_sp, device):
    """
    Đánh giá mô hình trên các câu tùy chỉnh
    
    Args:
        sentences (list): Danh sách câu cần dịch
        transformer_model: Mô hình Transformer
        lstm_model: Mô hình LSTM
        src_sp: Tokenizer nguồn
        tgt_sp: Tokenizer đích
        device (str): Thiết bị (CPU/GPU)
        
    Returns:
        pd.DataFrame: DataFrame chứa kết quả dịch
    """
    transformer_translations = []
    lstm_translations = []
    
    for sentence in tqdm(sentences, desc="Dịch câu tùy chỉnh"):
        # Dịch bằng Transformer
        transformer_translation = translate_sentence(
            sentence, transformer_model, src_sp, tgt_sp, 'transformer', device
        )
        
        # Dịch bằng LSTM
        lstm_translation = translate_sentence(
            sentence, lstm_model, src_sp, tgt_sp, 'lstm', device
        )
        
        transformer_translations.append(transformer_translation)
        lstm_translations.append(lstm_translation)
    
    # Tạo DataFrame
    results_df = pd.DataFrame({
        'Source': sentences,
        'Transformer': transformer_translations,
        'LSTM': lstm_translations
    })
    
    return results_df


def main(config):
    """
    Hàm chính để đánh giá và so sánh mô hình
    
    Args:
        config (dict): Cấu hình
    """
    # Tải mô hình Transformer
    transformer_model, en_sp, vi_sp = load_model(
        os.path.join(config['model_dir'], 'transformer_best.pt'),
        'transformer',
        config,
        config['device']
    )
    
    # Tải mô hình LSTM
    lstm_model, _, _ = load_model(
        os.path.join(config['model_dir'], 'lstm_best.pt'),
        'lstm',
        config,
        config['device']
    )
    
    # Đọc dữ liệu test
    test_data = pd.read_csv(os.path.join(config['data_dir'], 'test_data.csv'))
    
    # Đánh giá mô hình Transformer trên tập test
    transformer_bleu, transformer_translations, references, src_sentences = evaluate_on_test_set(
        transformer_model, test_data, en_sp, vi_sp, 'transformer', config['device'], num_samples=config['num_samples']
    )
    
    # Đánh giá mô hình LSTM trên tập test
    lstm_bleu, lstm_translations, _, _ = evaluate_on_test_set(
        lstm_model, test_data, en_sp, vi_sp, 'lstm', config['device'], num_samples=config['num_samples']
    )
    
    logger.info(f"BLEU score của mô hình Transformer: {transformer_bleu:.2f}")
    logger.info(f"BLEU score của mô hình LSTM: {lstm_bleu:.2f}")
    
    # So sánh kết quả dịch
    comparison_df = compare_translations(
        src_sentences, transformer_translations, lstm_translations, references,
        output_path=os.path.join(config['model_dir'], 'translation_comparison.csv')
    )
    
    # Trực quan hóa kết quả so sánh
    visualize_comparison(
        comparison_df,
        output_path=os.path.join(config['model_dir'], 'translation_comparison.png')
    )
    
    # Đánh giá trên các câu tùy chỉnh
    custom_sentences = [
        "Hello, how are you?",
        "I love learning new languages.",
        "The weather is beautiful today.",
        "Can you help me translate this document?",
        "I will visit Vietnam next year.",
        "This is a complex sentence with multiple clauses and technical terms.",
        "Artificial intelligence is transforming the way we live and work.",
        "Please send me the report by email as soon as possible.",
        "The conference will be held in Hanoi from June 10 to June 15.",
        "I don't understand what you're saying."
    ]
    
    custom_results = evaluate_on_custom_sentences(
        custom_sentences, transformer_model, lstm_model, en_sp, vi_sp, config['device']
    )
    
    # Lưu kết quả
    custom_results.to_csv(os.path.join(config['model_dir'], 'custom_translations.csv'), index=False)
    logger.info(f"Đã lưu kết quả dịch câu tùy chỉnh tại {os.path.join(config['model_dir'], 'custom_translations.csv')}")
    
    # In kết quả
    logger.info("\nKết quả dịch câu tùy chỉnh:")
    print(tabulate(custom_results, headers='keys', tablefmt='grid'))
    
    # Tạo báo cáo tổng hợp
    summary = {
        'Model': ['Transformer', 'LSTM'],
        'BLEU Score': [transformer_bleu, lstm_bleu],
        'Better on Test Set (%)': [
            comparison_df[comparison_df['Better Model'] == 'Transformer'].shape[0] / comparison_df.shape[0] * 100,
            comparison_df[comparison_df['Better Model'] == 'LSTM'].shape[0] / comparison_df.shape[0] * 100
        ]
    }
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(config['model_dir'], 'model_summary.csv'), index=False)
    logger.info(f"Đã lưu báo cáo tổng hợp tại {os.path.join(config['model_dir'], 'model_summary.csv')}")
    
    # In báo cáo tổng hợp
    logger.info("\nBáo cáo tổng hợp:")
    print(tabulate(summary_df, headers='keys', tablefmt='grid'))
    
    # Xác định mô hình tốt nhất
    best_model = 'Transformer' if transformer_bleu > lstm_bleu else 'LSTM'
    logger.info(f"\nMô hình tốt nhất: {best_model}")
    
    return best_model, transformer_bleu, lstm_bleu


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Đánh giá và so sánh mô hình dịch máy')
    parser.add_argument('--data_dir', type=str, default='./data', help='Thư mục dữ liệu')
    parser.add_argument('--model_dir', type=str, default='./models', help='Thư mục mô hình')
    parser.add_argument('--num_samples', type=int, default=100, help='Số lượng mẫu để đánh giá')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Thiết bị (CPU/GPU)')
    parser.add_argument('--src_lang', type=str, default='en', help='Ngôn ngữ nguồn')
    parser.add_argument('--tgt_lang', type=str, default='vi', help='Ngôn ngữ đích')
    args = parser.parse_args()
    
    # Cấu hình
    config = {
        'data_dir': args.data_dir,
        'model_dir': args.model_dir,
        'num_samples': args.num_samples,
        'device': args.device,
        'src_lang': args.src_lang,
        'tgt_lang': args.tgt_lang,
        
        'transformer': {
            'd_model': 512,
            'nhead': 8,
            'num_encoder_layers': 6,
            'num_decoder_layers': 6,
            'd_ff': 2048,
            'dropout': 0.1,
            'pad_idx': 0,
            'label_smoothing': 0.1
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
    
    # Đánh giá và so sánh mô hình
    best_model, transformer_bleu, lstm_bleu = main(config)
