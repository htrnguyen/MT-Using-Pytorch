#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mô hình dịch máy Transformer cho Tiếng Việt - Tiếng Anh
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pad_sequence


class PositionalEncoding(nn.Module):
    """
    Mã hóa vị trí cho Transformer
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Khởi tạo lớp mã hóa vị trí
        
        Args:
            d_model (int): Kích thước embedding
            max_len (int): Độ dài tối đa của câu
            dropout (float): Tỷ lệ dropout
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Tạo ma trận mã hóa vị trí
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor shape [batch_size, seq_len, d_model]
        
        Returns:
            Tensor shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """
    Encoder của Transformer
    """
    def __init__(self, src_vocab_size, d_model, nhead, num_encoder_layers, 
                 dim_feedforward, dropout, pad_idx):
        """
        Khởi tạo Encoder
        
        Args:
            src_vocab_size (int): Kích thước từ điển nguồn
            d_model (int): Kích thước embedding
            nhead (int): Số lượng head trong Multi-Head Attention
            num_encoder_layers (int): Số lượng lớp encoder
            dim_feedforward (int): Kích thước hidden layer trong feed-forward network
            dropout (float): Tỷ lệ dropout
            pad_idx (int): Chỉ số của token padding
        """
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )
        
        self.scale = math.sqrt(d_model)
        self.pad_idx = pad_idx
        
    def forward(self, src, src_mask=None):
        """
        Args:
            src: Tensor shape [batch_size, src_len]
            src_mask: Tensor shape [batch_size, 1, 1, src_len]
        
        Returns:
            Tensor shape [batch_size, src_len, d_model]
        """
        # Tạo mask cho padding
        if src_mask is None:
            src_key_padding_mask = (src == self.pad_idx)
        else:
            src_key_padding_mask = None
        
        # Embedding và mã hóa vị trí
        src = self.src_embedding(src) * self.scale
        src = self.pos_encoding(src)
        
        # Áp dụng Transformer Encoder
        memory = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        
        return memory


class TransformerDecoder(nn.Module):
    """
    Decoder của Transformer
    """
    def __init__(self, tgt_vocab_size, d_model, nhead, num_decoder_layers, 
                 dim_feedforward, dropout, pad_idx):
        """
        Khởi tạo Decoder
        
        Args:
            tgt_vocab_size (int): Kích thước từ điển đích
            d_model (int): Kích thước embedding
            nhead (int): Số lượng head trong Multi-Head Attention
            num_decoder_layers (int): Số lượng lớp decoder
            dim_feedforward (int): Kích thước hidden layer trong feed-forward network
            dropout (float): Tỷ lệ dropout
            pad_idx (int): Chỉ số của token padding
        """
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers
        )
        
        self.scale = math.sqrt(d_model)
        self.pad_idx = pad_idx
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Args:
            tgt: Tensor shape [batch_size, tgt_len]
            memory: Tensor shape [batch_size, src_len, d_model]
            tgt_mask: Tensor shape [tgt_len, tgt_len]
            memory_mask: Tensor shape [batch_size, 1, 1, src_len]
        
        Returns:
            Tensor shape [batch_size, tgt_len, d_model]
        """
        # Tạo mask cho padding
        tgt_key_padding_mask = (tgt == self.pad_idx)
        memory_key_padding_mask = None if memory_mask is None else memory_mask.squeeze(1).squeeze(1)
        
        # Embedding và mã hóa vị trí
        tgt = self.tgt_embedding(tgt) * self.scale
        tgt = self.pos_encoding(tgt)
        
        # Áp dụng Transformer Decoder
        output = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        return output


class TransformerModel(nn.Module):
    """
    Mô hình Transformer đầy đủ
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead, 
                 num_encoder_layers, num_decoder_layers, dim_feedforward, 
                 dropout, pad_idx):
        """
        Khởi tạo mô hình Transformer
        
        Args:
            src_vocab_size (int): Kích thước từ điển nguồn
            tgt_vocab_size (int): Kích thước từ điển đích
            d_model (int): Kích thước embedding
            nhead (int): Số lượng head trong Multi-Head Attention
            num_encoder_layers (int): Số lượng lớp encoder
            num_decoder_layers (int): Số lượng lớp decoder
            dim_feedforward (int): Kích thước hidden layer trong feed-forward network
            dropout (float): Tỷ lệ dropout
            pad_idx (int): Chỉ số của token padding
        """
        super(TransformerModel, self).__init__()
        
        self.pad_idx = pad_idx
        
        # Encoder
        self.encoder = TransformerEncoder(
            src_vocab_size=src_vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pad_idx=pad_idx
        )
        
        # Decoder
        self.decoder = TransformerDecoder(
            tgt_vocab_size=tgt_vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pad_idx=pad_idx
        )
        
        # Generator layer
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        
        # Khởi tạo trọng số
        self._reset_parameters()
        
    def _reset_parameters(self):
        """
        Khởi tạo trọng số cho mô hình
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def make_src_mask(self, src):
        """
        Tạo mask cho source sequence
        
        Args:
            src: Tensor shape [batch_size, src_len]
        
        Returns:
            Tensor shape [batch_size, 1, 1, src_len]
        """
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def make_tgt_mask(self, tgt):
        """
        Tạo mask cho target sequence
        
        Args:
            tgt: Tensor shape [batch_size, tgt_len]
        
        Returns:
            Tensor shape [tgt_len, tgt_len]
        """
        tgt_len = tgt.size(1)
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len) * float('-inf'), diagonal=1)
        tgt_mask = tgt_mask.to(tgt.device)
        return tgt_mask
    
    def forward(self, src, tgt):
        """
        Args:
            src: Tensor shape [batch_size, src_len]
            tgt: Tensor shape [batch_size, tgt_len]
        
        Returns:
            Tensor shape [batch_size, tgt_len, tgt_vocab_size]
        """
        # Tạo mask
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        
        # Encoder
        enc_output = self.encoder(src, src_mask)
        
        # Decoder
        dec_output = self.decoder(tgt, enc_output, tgt_mask, src_mask)
        
        # Generator
        output = self.generator(dec_output)
        
        return output


def create_transformer_model(config, src_vocab_size, tgt_vocab_size):
    """
    Tạo mô hình Transformer
    
    Args:
        config (dict): Cấu hình mô hình
        src_vocab_size (int): Kích thước từ điển nguồn
        tgt_vocab_size (int): Kích thước từ điển đích
    
    Returns:
        TransformerModel: Mô hình Transformer
    """
    model = TransformerModel(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        dim_feedforward=config['d_ff'],
        dropout=config['dropout'],
        pad_idx=config['pad_idx']
    )
    
    return model


def translate(model, src, src_sp, tgt_sp, device, max_len=100):
    """
    Dịch một câu
    
    Args:
        model (TransformerModel): Mô hình Transformer
        src (str): Câu nguồn
        src_sp: Tokenizer nguồn
        tgt_sp: Tokenizer đích
        device (str): Thiết bị (CPU/GPU)
        max_len (int): Độ dài tối đa của câu dịch
    
    Returns:
        str: Câu đã dịch
    """
    model.eval()
    
    # Tokenize câu nguồn
    src_tokens = [src_sp.bos_id()] + src_sp.encode(src, out_type=int) + [src_sp.eos_id()]
    src_tensor = torch.tensor(src_tokens).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)
    
    # Encoder
    with torch.no_grad():
        enc_output = model.encoder(src_tensor, src_mask)
    
    # Bắt đầu với token BOS
    tgt_tokens = [tgt_sp.bos_id()]
    tgt_tensor = torch.tensor([tgt_tokens]).to(device)
    
    # Dịch từng token
    for _ in range(max_len):
        with torch.no_grad():
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


class NoamOpt:
    """
    Optimizer với learning rate schedule theo công thức Noam
    """
    def __init__(self, model_size, factor, warmup, optimizer):
        """
        Khởi tạo optimizer
        
        Args:
            model_size (int): Kích thước mô hình (d_model)
            factor (float): Hệ số nhân
            warmup (int): Số bước warmup
            optimizer: Optimizer cơ bản
        """
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        """
        Cập nhật learning rate và thực hiện bước tối ưu
        """
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step=None):
        """
        Tính learning rate tại bước hiện tại
        
        Args:
            step (int, optional): Bước cụ thể, mặc định là bước hiện tại
        
        Returns:
            float: Learning rate
        """
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))
