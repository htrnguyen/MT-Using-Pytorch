#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tiền xử lý dữ liệu cho mô hình dịch máy Tiếng Việt - Tiếng Anh
"""

import pandas as pd
import numpy as np
import re
import unicodedata
import sentencepiece as spm
from sklearn.model_selection import train_test_split
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import logging

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, config):
        """
        Khởi tạo bộ tiền xử lý dữ liệu
        
        Args:
            config (dict): Cấu hình cho quá trình tiền xử lý
        """
        self.config = config
        self.en_sp = None
        self.vi_sp = None
        
        # Tạo thư mục lưu trữ dữ liệu
        os.makedirs(config['data_dir'], exist_ok=True)
        os.makedirs(config['model_dir'], exist_ok=True)
        
    def load_data(self, file1_path, file2_path):
        """
        Đọc và gộp dữ liệu từ hai file, làm sạch dữ liệu.
        
        Args:
            file1_path (str): Đường dẫn file CSV
            file2_path (str): Đường dẫn file TSV
        
        Returns:
            pd.DataFrame: DataFrame chứa cặp câu song ngữ
        """
        logger.info(f"Đọc dữ liệu từ {file1_path} và {file2_path}")
        
        # Đọc file CSV
        df1 = pd.read_csv(file1_path, header=0, names=['en', 'vi'])
        df1 = df1.dropna().drop_duplicates()
        logger.info(f"Đọc được {len(df1)} cặp câu từ file CSV")

        # Đọc file TSV
        df2 = pd.read_csv(file2_path, sep='\t', header=None, names=['en', 'vi', 'metadata'])
        df2 = df2[['en', 'vi']].dropna().drop_duplicates()
        logger.info(f"Đọc được {len(df2)} cặp câu từ file TSV")

        # Gộp dữ liệu và loại bỏ trùng lặp
        merged_df = pd.concat([df1, df2], ignore_index=True).drop_duplicates(subset=['en', 'vi'])
        logger.info(f"Tổng số cặp câu sau khi gộp và loại bỏ trùng lặp: {len(merged_df)}")
        
        return merged_df
    
    def normalize_text(self, text, is_vietnamese=False):
        """
        Chuẩn hóa văn bản: chuyển về chữ thường, loại bỏ khoảng trắng thừa,
        và xử lý đặc biệt cho tiếng Việt.
        
        Args:
            text (str): Văn bản cần chuẩn hóa
            is_vietnamese (bool): Có phải tiếng Việt không
            
        Returns:
            str: Văn bản đã chuẩn hóa
        """
        if not isinstance(text, str):
            return ""
        
        # Chuyển về chữ thường
        text = text.lower()
        
        # Chuẩn hóa Unicode cho tiếng Việt (NFC)
        if is_vietnamese:
            text = unicodedata.normalize('NFC', text)
        
        # Xử lý dấu câu: thêm khoảng trắng sau dấu câu nếu không có
        text = re.sub(r'([.,!?;:])([\w])', r'\1 \2', text)
        
        # Loại bỏ khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_data(self, data):
        """
        Tiền xử lý dữ liệu: chuẩn hóa, lọc câu quá dài hoặc quá ngắn
        
        Args:
            data (pd.DataFrame): DataFrame chứa cặp câu song ngữ
            
        Returns:
            pd.DataFrame: DataFrame đã tiền xử lý
        """
        logger.info("Bắt đầu tiền xử lý dữ liệu...")
        
        # Chuẩn hóa văn bản
        data['en_normalized'] = data['en'].apply(lambda x: self.normalize_text(x, is_vietnamese=False))
        data['vi_normalized'] = data['vi'].apply(lambda x: self.normalize_text(x, is_vietnamese=True))
        
        # Lọc câu quá ngắn
        min_len = self.config['min_sentence_length']
        data = data[(data['en_normalized'].str.split().str.len() >= min_len) & 
                    (data['vi_normalized'].str.split().str.len() >= min_len)]
        
        # Lọc câu quá dài
        max_len = self.config['max_sentence_length']
        data = data[(data['en_normalized'].str.split().str.len() <= max_len) & 
                    (data['vi_normalized'].str.split().str.len() <= max_len)]
        
        # Lọc tỷ lệ độ dài câu bất thường (có thể là câu không khớp)
        data['en_len'] = data['en_normalized'].str.split().str.len()
        data['vi_len'] = data['vi_normalized'].str.split().str.len()
        data['len_ratio'] = data['en_len'] / data['vi_len']
        
        # Lọc câu có tỷ lệ độ dài bất thường
        ratio_threshold = self.config['length_ratio_threshold']
        data = data[(data['len_ratio'] >= 1/ratio_threshold) & (data['len_ratio'] <= ratio_threshold)]
        
        logger.info(f"Số cặp câu sau khi lọc: {len(data)}")
        
        return data
    
    def visualize_data(self, data, save_path=None):
        """
        Trực quan hóa phân phối độ dài câu
        
        Args:
            data (pd.DataFrame): DataFrame chứa cặp câu
            save_path (str, optional): Đường dẫn để lưu biểu đồ
        """
        plt.figure(figsize=(12, 6))
        sns.histplot(data['en_len'], bins=30, color='blue', label='Tiếng Anh', alpha=0.6)
        sns.histplot(data['vi_len'], bins=30, color='orange', label='Tiếng Việt', alpha=0.6)
        plt.legend()
        plt.title('Phân phối độ dài câu trong dữ liệu')
        plt.xlabel('Số từ trong câu')
        plt.ylabel('Tần suất')
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Đã lưu biểu đồ tại {save_path}")
        else:
            plt.show()
    
    def train_tokenizers(self, data):
        """
        Huấn luyện tokenizer cho tiếng Anh và tiếng Việt
        
        Args:
            data (pd.DataFrame): DataFrame chứa cặp câu đã tiền xử lý
        """
        logger.info("Bắt đầu huấn luyện tokenizer...")
        
        # Lưu dữ liệu vào file tạm để huấn luyện tokenizer
        en_file = os.path.join(self.config['data_dir'], 'train_en.txt')
        vi_file = os.path.join(self.config['data_dir'], 'train_vi.txt')
        
        with open(en_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(data['en_normalized'].astype(str)))
        
        with open(vi_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(data['vi_normalized'].astype(str)))
        
        # Huấn luyện tokenizer cho tiếng Anh
        en_model_prefix = os.path.join(self.config['model_dir'], 'en_spm')
        spm.SentencePieceTrainer.train(
            f'--input={en_file} '
            f'--model_prefix={en_model_prefix} '
            f'--vocab_size={self.config["vocab_size"]} '
            f'--character_coverage=1.0 '
            f'--model_type={self.config["tokenizer_type"]} '
            f'--normalization_rule_name=nmt_nfkc '
            f'--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3'
        )
        
        # Huấn luyện tokenizer đặc biệt cho tiếng Việt với user_defined_symbols
        vi_model_prefix = os.path.join(self.config['model_dir'], 'vi_spm')
        
        # Tạo danh sách từ ghép tiếng Việt phổ biến
        vietnamese_compounds = self._extract_vietnamese_compounds(data['vi_normalized'])
        user_defined_symbols = ','.join(vietnamese_compounds)
        
        spm.SentencePieceTrainer.train(
            f'--input={vi_file} '
            f'--model_prefix={vi_model_prefix} '
            f'--vocab_size={self.config["vocab_size"]} '
            f'--character_coverage=1.0 '
            f'--model_type={self.config["tokenizer_type"]} '
            f'--normalization_rule_name=nmt_nfkc '
            f'--user_defined_symbols={user_defined_symbols} '
            f'--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3'
        )
        
        # Load tokenizer
        self.en_sp = spm.SentencePieceProcessor(model_file=f'{en_model_prefix}.model')
        self.vi_sp = spm.SentencePieceProcessor(model_file=f'{vi_model_prefix}.model')
        
        logger.info(f"Đã huấn luyện tokenizer với vocab_size={self.config['vocab_size']}")
    
    def _extract_vietnamese_compounds(self, texts, min_freq=5):
        """
        Trích xuất các từ ghép tiếng Việt phổ biến để đưa vào user_defined_symbols
        
        Args:
            texts (pd.Series): Chuỗi các câu tiếng Việt
            min_freq (int): Tần suất tối thiểu để coi là từ ghép phổ biến
            
        Returns:
            list: Danh sách các từ ghép tiếng Việt phổ biến
        """
        # Danh sách một số từ ghép tiếng Việt phổ biến
        predefined_compounds = [
            "hà nội", "thành phố hồ chí minh", "hồ chí minh", 
            "việt nam", "hoa kỳ", "trung quốc", "nhà hàng",
            "siêu thị", "trường học", "bệnh viện", "công ty",
            "thành phố", "quốc gia", "đất nước", "con người",
            "gia đình", "nhà cửa", "xe máy", "ô tô", "máy tính",
            "điện thoại", "internet", "mạng xã hội", "facebook",
            "google", "youtube", "instagram", "twitter", "tiktok"
        ]
        
        # Tìm các từ ghép xuất hiện nhiều trong dữ liệu
        word_pairs = {}
        for text in texts:
            words = text.split()
            for i in range(len(words) - 1):
                pair = f"{words[i]} {words[i+1]}"
                word_pairs[pair] = word_pairs.get(pair, 0) + 1
        
        # Lọc các từ ghép xuất hiện nhiều
        frequent_pairs = [pair for pair, count in word_pairs.items() if count >= min_freq]
        
        # Kết hợp với danh sách định nghĩa trước
        all_compounds = list(set(predefined_compounds + frequent_pairs))
        
        logger.info(f"Đã trích xuất {len(all_compounds)} từ ghép tiếng Việt phổ biến")
        return all_compounds
    
    def split_data(self, data):
        """
        Chia dữ liệu thành tập train, validation và test
        
        Args:
            data (pd.DataFrame): DataFrame chứa cặp câu đã tiền xử lý
            
        Returns:
            tuple: (train_data, val_data, test_data)
        """
        # Chia tập train và phần còn lại
        train_data, temp_data = train_test_split(
            data, 
            test_size=self.config['val_test_size'],
            random_state=self.config['random_seed']
        )
        
        # Chia phần còn lại thành validation và test
        val_data, test_data = train_test_split(
            temp_data,
            test_size=0.5,
            random_state=self.config['random_seed']
        )
        
        logger.info(f"Chia dữ liệu: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        return train_data, val_data, test_data
    
    def save_processed_data(self, train_data, val_data, test_data):
        """
        Lưu dữ liệu đã xử lý
        
        Args:
            train_data, val_data, test_data (pd.DataFrame): Các tập dữ liệu đã chia
        """
        data_dir = self.config['data_dir']
        
        train_data.to_csv(os.path.join(data_dir, 'train_data.csv'), index=False)
        val_data.to_csv(os.path.join(data_dir, 'val_data.csv'), index=False)
        test_data.to_csv(os.path.join(data_dir, 'test_data.csv'), index=False)
        
        # Lưu tokenizer
        with open(os.path.join(self.config['model_dir'], 'tokenizers.pkl'), 'wb') as f:
            pickle.dump({'en_sp': self.en_sp, 'vi_sp': self.vi_sp}, f)
        
        logger.info(f"Đã lưu dữ liệu đã xử lý vào {data_dir}")
    
    def process(self, file1_path, file2_path):
        """
        Thực hiện toàn bộ quy trình tiền xử lý
        
        Args:
            file1_path (str): Đường dẫn file CSV
            file2_path (str): Đường dẫn file TSV
            
        Returns:
            tuple: (train_data, val_data, test_data, en_sp, vi_sp)
        """
        # Đọc dữ liệu
        data = self.load_data(file1_path, file2_path)
        
        # Tiền xử lý dữ liệu
        processed_data = self.preprocess_data(data)
        
        # Trực quan hóa dữ liệu
        self.visualize_data(
            processed_data, 
            save_path=os.path.join(self.config['data_dir'], 'sentence_length_distribution.png')
        )
        
        # Huấn luyện tokenizer
        self.train_tokenizers(processed_data)
        
        # Chia dữ liệu
        train_data, val_data, test_data = self.split_data(processed_data)
        
        # Lưu dữ liệu đã xử lý
        self.save_processed_data(train_data, val_data, test_data)
        
        return train_data, val_data, test_data, self.en_sp, self.vi_sp


class TranslationDataset(Dataset):
    def __init__(self, data, en_sp, vi_sp, src_lang='en', tgt_lang='vi', max_len=None):
        """
        Dataset cho dữ liệu dịch máy
        
        Args:
            data (pd.DataFrame): DataFrame chứa cặp câu
            en_sp, vi_sp: Tokenizer cho tiếng Anh và tiếng Việt
            src_lang (str): Ngôn ngữ nguồn ('en' hoặc 'vi')
            tgt_lang (str): Ngôn ngữ đích ('vi' hoặc 'en')
            max_len (int, optional): Độ dài tối đa của câu
        """
        self.data = data
        self.en_sp = en_sp
        self.vi_sp = vi_sp
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_len = max_len
        
        # Xác định tokenizer cho ngôn ngữ nguồn và đích
        self.src_sp = self.en_sp if src_lang == 'en' else self.vi_sp
        self.tgt_sp = self.vi_sp if tgt_lang == 'vi' else self.en_sp
        
        # Xác định cột dữ liệu
        self.src_col = f'{src_lang}_normalized'
        self.tgt_col = f'{tgt_lang}_normalized'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text = str(self.data.iloc[idx][self.src_col])
        tgt_text = str(self.data.iloc[idx][self.tgt_col])
        
        # Tokenize và thêm BOS/EOS
        src_tokens = [self.src_sp.bos_id()] + self.src_sp.encode(src_text, out_type=int) + [self.src_sp.eos_id()]
        tgt_tokens = [self.tgt_sp.bos_id()] + self.tgt_sp.encode(tgt_text, out_type=int) + [self.tgt_sp.eos_id()]
        
        # Cắt bớt nếu vượt quá độ dài tối đa
        if self.max_len:
            src_tokens = src_tokens[:self.max_len]
            tgt_tokens = tgt_tokens[:self.max_len]
        
        return torch.tensor(src_tokens), torch.tensor(tgt_tokens)


def collate_fn(batch, pad_idx=0):
    """
    Hàm padding cho batch
    
    Args:
        batch: Batch dữ liệu
        pad_idx (int): Chỉ số của token padding
        
    Returns:
        tuple: (src_batch, tgt_batch, src_mask, tgt_mask)
    """
    src_batch, tgt_batch = zip(*batch)
    
    # Padding
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    
    # Tạo mask cho src (1 cho token thực, 0 cho padding)
    src_mask = (src_batch != pad_idx).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, src_len]
    
    # Tạo mask cho tgt (không nhìn được token trong tương lai)
    tgt_pad_mask = (tgt_batch != pad_idx).unsqueeze(1).unsqueeze(3)  # [batch_size, 1, tgt_len, 1]
    tgt_len = tgt_batch.size(1)
    tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt_batch.device)).bool()
    tgt_mask = tgt_pad_mask & tgt_sub_mask  # [batch_size, 1, tgt_len, tgt_len]
    
    return src_batch, tgt_batch, src_mask, tgt_mask


def create_dataloaders(train_data, val_data, test_data, en_sp, vi_sp, config):
    """
    Tạo DataLoader cho tập train, validation và test
    
    Args:
        train_data, val_data, test_data (pd.DataFrame): Các tập dữ liệu
        en_sp, vi_sp: Tokenizer cho tiếng Anh và tiếng Việt
        config (dict): Cấu hình
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Tạo dataset
    train_dataset = TranslationDataset(
        train_data, en_sp, vi_sp, 
        src_lang=config['src_lang'], 
        tgt_lang=config['tgt_lang'],
        max_len=config['max_len']
    )
    
    val_dataset = TranslationDataset(
        val_data, en_sp, vi_sp, 
        src_lang=config['src_lang'], 
        tgt_lang=config['tgt_lang'],
        max_len=config['max_len']
    )
    
    test_dataset = TranslationDataset(
        test_data, en_sp, vi_sp, 
        src_lang=config['src_lang'], 
        tgt_lang=config['tgt_lang'],
        max_len=config['max_len']
    )
    
    # Tạo dataloader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        collate_fn=lambda batch: collate_fn(batch, pad_idx=0),
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        collate_fn=lambda batch: collate_fn(batch, pad_idx=0),
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        collate_fn=lambda batch: collate_fn(batch, pad_idx=0),
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Cấu hình mặc định
    config = {
        'data_dir': './data',
        'model_dir': './models',
        'vocab_size': 8000,
        'tokenizer_type': 'bpe',  # 'bpe' hoặc 'unigram'
        'min_sentence_length': 3,
        'max_sentence_length': 100,
        'length_ratio_threshold': 2.5,
        'val_test_size': 0.2,
        'random_seed': 42,
        'src_lang': 'en',
        'tgt_lang': 'vi',
        'batch_size': 32,
        'max_len': 128,
        'num_workers': 2
    }
    
    # Đường dẫn file dữ liệu
    file1_path = './eng_vie.csv'
    file2_path = './vie.txt'
    
    # Tiền xử lý dữ liệu
    preprocessor = DataPreprocessor(config)
    train_data, val_data, test_data, en_sp, vi_sp = preprocessor.process(file1_path, file2_path)
    
    # Tạo dataloader
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data, en_sp, vi_sp, config
    )
    
    logger.info("Hoàn thành tiền xử lý dữ liệu!")
