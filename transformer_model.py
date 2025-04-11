#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mô hình Transformer cải tiến cho dịch máy Tiếng Việt - Tiếng Anh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy
import logging
from torch.nn.utils.rnn import pad_sequence

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """
    Mã hóa vị trí cho mô hình Transformer
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Khởi tạo mã hóa vị trí
        
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
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        # Đăng ký buffer (không phải parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x (Tensor): Embedding tensor [batch_size, seq_len, d_model]
            
        Returns:
            Tensor: Embedding với mã hóa vị trí [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention cải tiến với gradient checkpointing
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        """
        Khởi tạo Multi-Head Attention
        
        Args:
            d_model (int): Kích thước embedding
            nhead (int): Số lượng head
            dropout (float): Tỷ lệ dropout
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % nhead == 0, "d_model phải chia hết cho nhead"
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        # Linear layers cho Q, K, V và output
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query (Tensor): Query tensor [batch_size, q_len, d_model]
            key (Tensor): Key tensor [batch_size, k_len, d_model]
            value (Tensor): Value tensor [batch_size, v_len, d_model]
            mask (Tensor, optional): Mask tensor [batch_size, 1, q_len, k_len]
            
        Returns:
            Tensor: Output tensor [batch_size, q_len, d_model]
        """
        batch_size = query.size(0)
        
        # Linear projections và reshape
        q = self.wq(query).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)  # [batch_size, nhead, q_len, d_k]
        k = self.wk(key).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)    # [batch_size, nhead, k_len, d_k]
        v = self.wv(value).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)  # [batch_size, nhead, v_len, d_k]
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # [batch_size, nhead, q_len, k_len]
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)  # [batch_size, nhead, q_len, d_k]
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # [batch_size, q_len, d_model]
        
        return self.wo(output)


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Khởi tạo FFN
        
        Args:
            d_model (int): Kích thước embedding
            d_ff (int): Kích thước hidden layer
            dropout (float): Tỷ lệ dropout
        """
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # GELU thay vì ReLU
        
    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Tensor: Output tensor [batch_size, seq_len, d_model]
        """
        return self.fc2(self.dropout(self.activation(self.fc1(x))))


class EncoderLayer(nn.Module):
    """
    Encoder Layer của Transformer
    """
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        """
        Khởi tạo Encoder Layer
        
        Args:
            d_model (int): Kích thước embedding
            nhead (int): Số lượng head trong Multi-Head Attention
            d_ff (int): Kích thước hidden layer trong FFN
            dropout (float): Tỷ lệ dropout
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x (Tensor): Input tensor [batch_size, seq_len, d_model]
            mask (Tensor, optional): Mask tensor [batch_size, 1, 1, seq_len]
            
        Returns:
            Tensor: Output tensor [batch_size, seq_len, d_model]
        """
        # Self-Attention với residual connection và layer normalization
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-Forward với residual connection và layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """
    Decoder Layer của Transformer
    """
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        """
        Khởi tạo Decoder Layer
        
        Args:
            d_model (int): Kích thước embedding
            nhead (int): Số lượng head trong Multi-Head Attention
            d_ff (int): Kích thước hidden layer trong FFN
            dropout (float): Tỷ lệ dropout
        """
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        """
        Args:
            x (Tensor): Input tensor [batch_size, tgt_len, d_model]
            enc_output (Tensor): Encoder output [batch_size, src_len, d_model]
            tgt_mask (Tensor, optional): Target mask [batch_size, 1, tgt_len, tgt_len]
            src_mask (Tensor, optional): Source mask [batch_size, 1, 1, src_len]
            
        Returns:
            Tensor: Output tensor [batch_size, tgt_len, d_model]
        """
        # Self-Attention
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # Cross-Attention
        cross_attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-Forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class Encoder(nn.Module):
    """
    Encoder của Transformer
    """
    def __init__(self, vocab_size, d_model, nhead, d_ff, num_layers, dropout=0.1, max_len=5000):
        """
        Khởi tạo Encoder
        
        Args:
            vocab_size (int): Kích thước từ điển
            d_model (int): Kích thước embedding
            nhead (int): Số lượng head trong Multi-Head Attention
            d_ff (int): Kích thước hidden layer trong FFN
            num_layers (int): Số lượng encoder layer
            dropout (float): Tỷ lệ dropout
            max_len (int): Độ dài tối đa của câu
        """
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.scale = math.sqrt(d_model)
        
    def forward(self, x, mask=None):
        """
        Args:
            x (Tensor): Input tensor [batch_size, src_len]
            mask (Tensor, optional): Mask tensor [batch_size, 1, 1, src_len]
            
        Returns:
            Tensor: Output tensor [batch_size, src_len, d_model]
        """
        # Embedding và positional encoding
        x = self.embedding(x) * self.scale
        x = self.pos_encoding(x)
        
        # Encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)


class Decoder(nn.Module):
    """
    Decoder của Transformer
    """
    def __init__(self, vocab_size, d_model, nhead, d_ff, num_layers, dropout=0.1, max_len=5000):
        """
        Khởi tạo Decoder
        
        Args:
            vocab_size (int): Kích thước từ điển
            d_model (int): Kích thước embedding
            nhead (int): Số lượng head trong Multi-Head Attention
            d_ff (int): Kích thước hidden layer trong FFN
            num_layers (int): Số lượng decoder layer
            dropout (float): Tỷ lệ dropout
            max_len (int): Độ dài tối đa của câu
        """
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.scale = math.sqrt(d_model)
        
    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        """
        Args:
            x (Tensor): Input tensor [batch_size, tgt_len]
            enc_output (Tensor): Encoder output [batch_size, src_len, d_model]
            tgt_mask (Tensor, optional): Target mask [batch_size, 1, tgt_len, tgt_len]
            src_mask (Tensor, optional): Source mask [batch_size, 1, 1, src_len]
            
        Returns:
            Tensor: Output tensor [batch_size, tgt_len, d_model]
        """
        # Embedding và positional encoding
        x = self.embedding(x) * self.scale
        x = self.pos_encoding(x)
        
        # Decoder layers
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, src_mask)
        
        return self.norm(x)


class TransformerModel(nn.Module):
    """
    Mô hình Transformer cho dịch máy
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, 
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, 
                 dropout=0.1, max_len=5000, pad_idx=0, label_smoothing=0.1):
        """
        Khởi tạo mô hình Transformer
        
        Args:
            src_vocab_size (int): Kích thước từ điển nguồn
            tgt_vocab_size (int): Kích thước từ điển đích
            d_model (int): Kích thước embedding
            nhead (int): Số lượng head trong Multi-Head Attention
            num_encoder_layers (int): Số lượng encoder layer
            num_decoder_layers (int): Số lượng decoder layer
            d_ff (int): Kích thước hidden layer trong FFN
            dropout (float): Tỷ lệ dropout
            max_len (int): Độ dài tối đa của câu
            pad_idx (int): Chỉ số của token padding
            label_smoothing (float): Hệ số label smoothing
        """
        super(TransformerModel, self).__init__()
        self.pad_idx = pad_idx
        self.encoder = Encoder(src_vocab_size, d_model, nhead, d_ff, num_encoder_layers, dropout, max_len)
        self.decoder = Decoder(tgt_vocab_size, d_model, nhead, d_ff, num_decoder_layers, dropout, max_len)
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        
        # Label smoothing
        self.criterion = LabelSmoothingLoss(tgt_vocab_size, padding_idx=pad_idx, smoothing=label_smoothing)
        
        # Khởi tạo trọng số
        self._init_parameters()
        
    def _init_parameters(self):
        """
        Khởi tạo trọng số của mô hình
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Args:
            src (Tensor): Source tensor [batch_size, src_len]
            tgt (Tensor): Target tensor [batch_size, tgt_len]
            src_mask (Tensor, optional): Source mask [batch_size, 1, 1, src_len]
            tgt_mask (Tensor, optional): Target mask [batch_size, 1, tgt_len, tgt_len]
            
        Returns:
            Tensor: Output tensor [batch_size, tgt_len, tgt_vocab_size]
        """
        # Tạo mask nếu chưa có
        if src_mask is None:
            src_mask = self.make_src_mask(src)
        if tgt_mask is None:
            tgt_mask = self.make_tgt_mask(tgt)
        
        # Encoder
        enc_output = self.encoder(src, src_mask)
        
        # Decoder
        dec_output = self.decoder(tgt, enc_output, tgt_mask, src_mask)
        
        # Generator
        output = self.generator(dec_output)
        
        return output
    
    def make_src_mask(self, src):
        """
        Tạo mask cho source sequence
        
        Args:
            src (Tensor): Source tensor [batch_size, src_len]
            
        Returns:
            Tensor: Source mask [batch_size, 1, 1, src_len]
        """
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def make_tgt_mask(self, tgt):
        """
        Tạo mask cho target sequence
        
        Args:
            tgt (Tensor): Target tensor [batch_size, tgt_len]
            
        Returns:
            Tensor: Target mask [batch_size, 1, tgt_len, tgt_len]
        """
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(3)
        tgt_len = tgt.size(1)
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return tgt_mask
    
    def count_parameters(self):
        """
        Đếm số lượng tham số của mô hình
        
        Returns:
            int: Số lượng tham số
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def calculate_loss(self, output, target):
        """
        Tính loss
        
        Args:
            output (Tensor): Output tensor [batch_size, tgt_len, tgt_vocab_size]
            target (Tensor): Target tensor [batch_size, tgt_len]
            
        Returns:
            Tensor: Loss
        """
        # Reshape output và target
        batch_size, tgt_len, tgt_vocab_size = output.size()
        output = output.contiguous().view(-1, tgt_vocab_size)
        target = target.contiguous().view(-1)
        
        # Tính loss
        return self.criterion(output, target)


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss
    """
    def __init__(self, size, padding_idx, smoothing=0.0):
        """
        Khởi tạo Label Smoothing Loss
        
        Args:
            size (int): Kích thước từ điển
            padding_idx (int): Chỉ số của token padding
            smoothing (float): Hệ số smoothing
        """
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        """
        Args:
            x (Tensor): Logits [batch_size, vocab_size]
            target (Tensor): Target [batch_size]
            
        Returns:
            Tensor: Loss
        """
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(F.log_softmax(x, dim=1), true_dist) / x.size(0)


class NoamOpt:
    """
    Optimizer với learning rate schedule theo công thức Noam
    """
    def __init__(self, model_size, factor, warmup, optimizer):
        """
        Khởi tạo Noam Optimizer
        
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
            step (int, optional): Bước hiện tại, mặc định là self._step
            
        Returns:
            float: Learning rate
        """
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))
    
    def zero_grad(self):
        """
        Xóa gradient
        """
        self.optimizer.zero_grad()


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
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        max_len=config['max_len'],
        pad_idx=config['pad_idx'],
        label_smoothing=config['label_smoothing']
    )
    
    logger.info(f"Đã tạo mô hình Transformer với {model.count_parameters():,} tham số")
    
    return model


def translate(model, src, src_sp, tgt_sp, device, max_len=100):
    """
    Dịch một câu từ ngôn ngữ nguồn sang ngôn ngữ đích
    
    Args:
        model (TransformerModel): Mô hình Transformer
        src (str): Câu nguồn
        src_sp: Tokenizer nguồn
        tgt_sp: Tokenizer đích
        device: Thiết bị (CPU/GPU)
        max_len (int): Độ dài tối đa của câu dịch
        
    Returns:
        str: Câu dịch
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
    tgt_tensor = torch.tensor(tgt_tokens).unsqueeze(0).to(device)
    
    # Dịch từng token
    for i in range(max_len):
        tgt_mask = model.make_tgt_mask(tgt_tensor)
        
        with torch.no_grad():
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
    tgt_tokens = tgt_tokens[1:]  # Bỏ token BOS
    if tgt_tokens[-1] == tgt_sp.eos_id():
        tgt_tokens = tgt_tokens[:-1]  # Bỏ token EOS
    
    return tgt_sp.decode(tgt_tokens)


if __name__ == "__main__":
    # Cấu hình mặc định
    config = {
        'd_model': 512,
        'nhead': 8,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'd_ff': 2048,
        'dropout': 0.1,
        'max_len': 128,
        'pad_idx': 0,
        'label_smoothing': 0.1
    }
    
    # Tạo mô hình
    model = create_transformer_model(config, src_vocab_size=8000, tgt_vocab_size=8000)
    logger.info(f"Số lượng tham số: {model.count_parameters():,}")
