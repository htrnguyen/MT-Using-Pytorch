#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tiền xử lý dữ liệu cho mô hình dịch máy Tiếng Việt - Tiếng Anh
"""

import os
import re
import unicodedata
import pandas as pd
import numpy as np
import torch
import logging
import pickle
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Lớp tiền xử lý dữ liệu cho mô hình dịch máy
    """
    def __init__(self, config):
        """
        Khởi tạo DataPreprocessor
        
        Args:
            config (dict): Cấu hình tiền xử lý
        """
        self.config = config
        
        # Tạo thư mục
        os.makedirs(config['data_dir'], exist_ok=True)
        os.makedirs(config['model_dir'], exist_ok=True)
        
        # Từ đặc biệt cho tiếng Việt
        self.vi_special_words = [
            'Hà Nội', 'Sài Gòn', 'Hồ Chí Minh', 'Việt Nam', 'Huế', 'Đà Nẵng', 'Cần Thơ',
            'Hải Phòng', 'Nha Trang', 'Vũng Tàu', 'Đà Lạt', 'Phú Quốc', 'Hạ Long',
            'Tây Nguyên', 'Mekong', 'Hồng', 'Cửu Long', 'Trường Sa', 'Hoàng Sa',
            'Tết', 'Âm lịch', 'Dương lịch', 'Nguyễn', 'Trần', 'Lê', 'Phạm', 'Hoàng',
            'Huỳnh', 'Phan', 'Vũ', 'Võ', 'Đặng', 'Bùi', 'Đỗ', 'Hồ', 'Ngô', 'Dương', 'Lý'
        ]
    
    def normalize_unicode(self, text):
        """
        Chuẩn hóa Unicode cho tiếng Việt
        
        Args:
            text (str): Văn bản cần chuẩn hóa
        
        Returns:
            str: Văn bản đã chuẩn hóa
        """
        return unicodedata.normalize('NFC', text)
    
    def clean_text(self, text, is_vietnamese=False):
        """
        Làm sạch văn bản
        
        Args:
            text (str): Văn bản cần làm sạch
            is_vietnamese (bool): Có phải tiếng Việt không
        
        Returns:
            str: Văn bản đã làm sạch
        """
        # Chuẩn hóa Unicode
        text = self.normalize_unicode(text)
        
        # Loại bỏ khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Xử lý đặc biệt cho tiếng Việt
        if is_vietnamese:
            # Bảo vệ các từ đặc biệt
            for word in self.vi_special_words:
                if word in text:
                    # Thay thế tạm thời để bảo vệ
                    text = text.replace(word, f"__{word.replace(' ', '_')}__")
            
            # Sau khi xử lý, khôi phục lại
            text = re.sub(r'__(.+?)__', lambda m: m.group(1).replace('_', ' '), text)
        
        return text
    
    def load_data(self, file1_path, file2_path):
        """
        Tải dữ liệu từ file
        
        Args:
            file1_path (str): Đường dẫn đến file CSV
            file2_path (str): Đường dẫn đến file TSV
        
        Returns:
            pd.DataFrame: DataFrame chứa cặp câu tiếng Anh - tiếng Việt
        """
        logger.info("Tải dữ liệu từ file...")
        
        # Tải dữ liệu từ file CSV
        df1 = pd.read_csv(file1_path)
        
        # Tải dữ liệu từ file TSV
        with open(file2_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        en_sentences = []
        vi_sentences = []
        
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                en_sentences.append(parts[0])
                vi_sentences.append(parts[1])
        
        df2 = pd.DataFrame({
            'en': en_sentences,
            'vi': vi_sentences
        })
        
        # Kết hợp hai DataFrame
        df = pd.concat([df1, df2], ignore_index=True)
        
        # Loại bỏ các dòng trùng lặp
        df = df.drop_duplicates().reset_index(drop=True)
        
        logger.info(f"Đã tải {len(df)} cặp câu")
        
        return df
    
    def preprocess_data(self, data):
        """
        Tiền xử lý dữ liệu
        
        Args:
            data (pd.DataFrame): DataFrame chứa cặp câu tiếng Anh - tiếng Việt
        
        Returns:
            pd.DataFrame: DataFrame chứa cặp câu đã tiền xử lý
        """
        logger.info("Tiền xử lý dữ liệu...")
        
        # Làm sạch văn bản
        data['en_normalized'] = data['en'].apply(lambda x: self.clean_text(x))
        data['vi_normalized'] = data['vi'].apply(lambda x: self.clean_text(x, is_vietnamese=True))
        
        # Tính độ dài câu
        data['en_length'] = data['en_normalized'].apply(len)
        data['vi_length'] = data['vi_normalized'].apply(len)
        
        # Tính tỷ lệ độ dài
        data['length_ratio'] = data['vi_length'] / data['en_length']
        
        # Lọc câu quá ngắn hoặc quá dài
        min_len = self.config['min_sentence_length']
        max_len = self.config['max_sentence_length']
        ratio_threshold = self.config['length_ratio_threshold']
        
        filtered_data = data[
            (data['en_length'] >= min_len) & 
            (data['en_length'] <= max_len) &
            (data['vi_length'] >= min_len) & 
            (data['vi_length'] <= max_len) &
            (data['length_ratio'] <= ratio_threshold) &
            (data['length_ratio'] >= 1/ratio_threshold)
        ]
        
        logger.info(f"Sau khi lọc: {len(filtered_data)} cặp câu")
        
        return filtered_data
    
    def train_tokenizers(self, data):
        """
        Huấn luyện tokenizer
        
        Args:
            data (pd.DataFrame): DataFrame chứa cặp câu đã tiền xử lý
        
        Returns:
            tuple: (en_sp, vi_sp) - Các tokenizer đã huấn luyện
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
        
        # Tạo danh sách user_defined_symbols từ vi_special_words
        user_defined_symbols = ','.join(self.vi_special_words)
        
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
        
        # Tải tokenizer
        en_sp = spm.SentencePieceProcessor()
        vi_sp = spm.SentencePieceProcessor()
        
        en_sp.load(f'{en_model_prefix}.model')
        vi_sp.load(f'{vi_model_prefix}.model')
        
        # Lưu tokenizer
        tokenizers = {
            'en_sp': en_sp,
            'vi_sp': vi_sp
        }
        
        with open(os.path.join(self.config['model_dir'], 'tokenizers.pkl'), 'wb') as f:
            pickle.dump(tokenizers, f)
        
        logger.info("Đã huấn luyện tokenizer thành công")
        
        return en_sp, vi_sp
    
    def split_data(self, data):
        """
        Chia dữ liệu thành tập train, validation và test
        
        Args:
            data (pd.DataFrame): DataFrame chứa cặp câu đã tiền xử lý
        
        Returns:
            tuple: (train_data, val_data, test_data) - Các tập dữ liệu đã chia
        """
        logger.info("Chia dữ liệu thành tập train, validation và test...")
        
        # Chia dữ liệu
        train_data, temp_data = train_test_split(
            data, 
            test_size=self.config['val_test_size'], 
            random_state=self.config['random_seed']
        )
        
        val_data, test_data = train_test_split(
            temp_data, 
            test_size=0.5, 
            random_state=self.config['random_seed']
        )
        
        # Lưu dữ liệu
        train_data.to_csv(os.path.join(self.config['data_dir'], 'train_data.csv'), index=False)
        val_data.to_csv(os.path.join(self.config['data_dir'], 'val_data.csv'), index=False)
        test_data.to_csv(os.path.join(self.config['data_dir'], 'test_data.csv'), index=False)
        
        logger.info(f"Tập train: {len(train_data)} cặp câu")
        logger.info(f"Tập validation: {len(val_data)} cặp câu")
        logger.info(f"Tập test: {len(test_data)} cặp câu")
        
        return train_data, val_data, test_data
    
    def process(self, file1_path, file2_path):
        """
        Xử lý dữ liệu từ đầu đến cuối
        
        Args:
            file1_path (str): Đường dẫn đến file CSV
            file2_path (str): Đường dẫn đến file TSV
        
        Returns:
            tuple: (train_data, val_data, test_data, en_sp, vi_sp) - Dữ liệu và tokenizer
        """
        # Tải dữ liệu
        data = self.load_data(file1_path, file2_path)
        
        # Tiền xử lý dữ liệu
        data = self.preprocess_data(data)
        
        # Huấn luyện tokenizer
        en_sp, vi_sp = self.train_tokenizers(data)
        
        # Chia dữ liệu
        train_data, val_data, test_data = self.split_data(data)
        
        return train_data, val_data, test_data, en_sp, vi_sp


class TranslationDataset(Dataset):
    """
    Dataset cho mô hình dịch máy
    """
    def __init__(self, data, src_sp, tgt_sp, src_lang='en', tgt_lang='vi', max_len=128):
        """
        Khởi tạo Dataset
        
        Args:
            data (pd.DataFrame): DataFrame chứa cặp câu
            src_sp: Tokenizer nguồn
            tgt_sp: Tokenizer đích
            src_lang (str): Ngôn ngữ nguồn
            tgt_lang (str): Ngôn ngữ đích
            max_len (int): Độ dài tối đa của câu
        """
        self.data = data
        self.src_sp = src_sp
        self.tgt_sp = tgt_sp
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_len = max_len
        
        # Xác định cột dữ liệu
        self.src_col = f'{src_lang}_normalized'
        self.tgt_col = f'{tgt_lang}_normalized'
    
    def __len__(self):
        """
        Trả về số lượng cặp câu
        
        Returns:
            int: Số lượng cặp câu
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Lấy cặp câu tại vị trí idx
        
        Args:
            idx (int): Vị trí cặp câu
        
        Returns:
            tuple: (src_tokens, tgt_tokens, src_len, tgt_len) - Các token và độ dài
        """
        # Lấy câu
        src_text = self.data.iloc[idx][self.src_col]
        tgt_text = self.data.iloc[idx][self.tgt_col]
        
        # Tokenize
        src_tokens = [self.src_sp.bos_id()] + self.src_sp.encode(src_text, out_type=int) + [self.src_sp.eos_id()]
        tgt_tokens = [self.tgt_sp.bos_id()] + self.tgt_sp.encode(tgt_text, out_type=int) + [self.tgt_sp.eos_id()]
        
        # Cắt bớt nếu quá dài
        src_tokens = src_tokens[:self.max_len]
        tgt_tokens = tgt_tokens[:self.max_len]
        
        # Tính độ dài
        src_len = len(src_tokens)
        tgt_len = len(tgt_tokens)
        
        return src_tokens, tgt_tokens, src_len, tgt_len


def collate_fn(batch):
    """
    Hàm collate cho DataLoader
    
    Args:
        batch (list): Batch dữ liệu
    
    Returns:
        tuple: (src, tgt, src_lengths, tgt_lengths) - Các tensor đã padding
    """
    # Tách dữ liệu
    src_tokens, tgt_tokens, src_lengths, tgt_lengths = zip(*batch)
    
    # Chuyển sang tensor
    src_lengths = torch.tensor(src_lengths)
    tgt_lengths = torch.tensor(tgt_lengths)
    
    # Padding
    src = pad_sequence([torch.tensor(x) for x in src_tokens], batch_first=True, padding_value=0)
    tgt = pad_sequence([torch.tensor(x) for x in tgt_tokens], batch_first=True, padding_value=0)
    
    return src, tgt, src_lengths, tgt_lengths


def create_dataloaders(train_data, val_data, test_data, en_sp, vi_sp, config):
    """
    Tạo DataLoader cho các tập dữ liệu
    
    Args:
        train_data (pd.DataFrame): Tập train
        val_data (pd.DataFrame): Tập validation
        test_data (pd.DataFrame): Tập test
        en_sp: Tokenizer tiếng Anh
        vi_sp: Tokenizer tiếng Việt
        config (dict): Cấu hình
    
    Returns:
        tuple: (train_loader, val_loader, test_loader) - Các DataLoader
    """
    # Tạo Dataset
    train_dataset = TranslationDataset(
        train_data, 
        en_sp if config['src_lang'] == 'en' else vi_sp,
        vi_sp if config['tgt_lang'] == 'vi' else en_sp,
        src_lang=config['src_lang'],
        tgt_lang=config['tgt_lang'],
        max_len=config['max_len']
    )
    
    val_dataset = TranslationDataset(
        val_data, 
        en_sp if config['src_lang'] == 'en' else vi_sp,
        vi_sp if config['tgt_lang'] == 'vi' else en_sp,
        src_lang=config['src_lang'],
        tgt_lang=config['tgt_lang'],
        max_len=config['max_len']
    )
    
    test_dataset = TranslationDataset(
        test_data, 
        en_sp if config['src_lang'] == 'en' else vi_sp,
        vi_sp if config['tgt_lang'] == 'vi' else en_sp,
        src_lang=config['src_lang'],
        tgt_lang=config['tgt_lang'],
        max_len=config['max_len']
    )
    
    # Tạo DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
