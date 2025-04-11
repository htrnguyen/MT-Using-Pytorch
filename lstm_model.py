#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mô hình LSTM với Attention cho dịch máy Tiếng Việt - Tiếng Anh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import logging
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    """
    Encoder LSTM hai chiều
    """
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.3, bidirectional=True):
        """
        Khởi tạo Encoder
        
        Args:
            vocab_size (int): Kích thước từ điển nguồn
            embed_size (int): Kích thước embedding
            hidden_size (int): Kích thước hidden state
            num_layers (int): Số lớp LSTM
            dropout (float): Tỷ lệ dropout
            bidirectional (bool): Có sử dụng LSTM hai chiều không
        """
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Projection layer nếu sử dụng LSTM hai chiều
        if bidirectional:
            self.projection = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, src, src_lengths):
        """
        Args:
            src (Tensor): Source tensor [batch_size, src_len]
            src_lengths (Tensor): Độ dài thực của các câu [batch_size]
            
        Returns:
            tuple: (outputs, hidden, cell)
                - outputs: Tensor [batch_size, src_len, hidden_size * num_directions]
                - hidden: Tensor [num_layers * num_directions, batch_size, hidden_size]
                - cell: Tensor [num_layers * num_directions, batch_size, hidden_size]
        """
        # Embedding
        embedded = self.embedding(src)  # [batch_size, src_len, embed_size]
        
        # Pack sequence để tối ưu tính toán
        packed = pack_padded_sequence(embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # LSTM
        outputs, (hidden, cell) = self.lstm(packed)
        
        # Unpack sequence
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)  # [batch_size, src_len, hidden_size * num_directions]
        
        # Xử lý hidden state và cell state cho decoder
        if self.bidirectional:
            # Gộp hidden state từ hai chiều
            hidden = self._reshape_hidden(hidden)
            cell = self._reshape_hidden(cell)
        
        return outputs, hidden, cell
    
    def _reshape_hidden(self, hidden):
        """
        Reshape hidden state từ LSTM hai chiều
        
        Args:
            hidden (Tensor): Hidden state [num_layers * num_directions, batch_size, hidden_size]
            
        Returns:
            Tensor: Reshaped hidden state [num_layers, batch_size, hidden_size]
        """
        batch_size = hidden.size(1)
        
        # Reshape và gộp các chiều
        hidden = hidden.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
        
        # Gộp forward và backward
        hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)
        
        # Project về kích thước hidden_size
        hidden = self.projection(hidden.view(-1, self.hidden_size * 2)).view(self.num_layers, batch_size, self.hidden_size)
        
        return hidden


class Attention(nn.Module):
    """
    Cơ chế Attention cho Decoder
    """
    def __init__(self, enc_hidden_size, dec_hidden_size, attn_size=None):
        """
        Khởi tạo Attention
        
        Args:
            enc_hidden_size (int): Kích thước hidden state của encoder
            dec_hidden_size (int): Kích thước hidden state của decoder
            attn_size (int, optional): Kích thước attention, mặc định bằng dec_hidden_size
        """
        super(Attention, self).__init__()
        
        self.attn_size = attn_size if attn_size else dec_hidden_size
        
        # Projection layers
        self.enc_projection = nn.Linear(enc_hidden_size, self.attn_size, bias=False)
        self.dec_projection = nn.Linear(dec_hidden_size, self.attn_size, bias=False)
        self.out = nn.Linear(self.attn_size, 1, bias=False)
        
    def forward(self, decoder_hidden, encoder_outputs, src_mask=None):
        """
        Args:
            decoder_hidden (Tensor): Hidden state của decoder [batch_size, dec_hidden_size]
            encoder_outputs (Tensor): Outputs của encoder [batch_size, src_len, enc_hidden_size]
            src_mask (Tensor, optional): Mask cho source sequence [batch_size, src_len]
            
        Returns:
            tuple: (attention_weights, context_vector)
                - attention_weights: Tensor [batch_size, src_len]
                - context_vector: Tensor [batch_size, enc_hidden_size]
        """
        src_len = encoder_outputs.size(1)
        batch_size = encoder_outputs.size(0)
        
        # Reshape decoder hidden để tính attention
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)  # [batch_size, src_len, dec_hidden_size]
        
        # Project encoder outputs và decoder hidden
        enc_proj = self.enc_projection(encoder_outputs)  # [batch_size, src_len, attn_size]
        dec_proj = self.dec_projection(decoder_hidden)   # [batch_size, src_len, attn_size]
        
        # Tính energy
        energy = torch.tanh(enc_proj + dec_proj)  # [batch_size, src_len, attn_size]
        energy = self.out(energy).squeeze(2)      # [batch_size, src_len]
        
        # Áp dụng mask nếu có
        if src_mask is not None:
            energy = energy.masked_fill(src_mask == 0, -1e10)
        
        # Tính attention weights
        attention_weights = F.softmax(energy, dim=1)  # [batch_size, src_len]
        
        # Tính context vector
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # [batch_size, enc_hidden_size]
        
        return attention_weights, context_vector


class Decoder(nn.Module):
    """
    Decoder LSTM với Attention
    """
    def __init__(self, vocab_size, embed_size, hidden_size, enc_hidden_size=None, num_layers=2, dropout=0.3):
        """
        Khởi tạo Decoder
        
        Args:
            vocab_size (int): Kích thước từ điển đích
            embed_size (int): Kích thước embedding
            hidden_size (int): Kích thước hidden state
            enc_hidden_size (int, optional): Kích thước hidden state của encoder, mặc định bằng hidden_size
            num_layers (int): Số lớp LSTM
            dropout (float): Tỷ lệ dropout
        """
        super(Decoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.enc_hidden_size = enc_hidden_size if enc_hidden_size else hidden_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        
        # Attention
        self.attention = Attention(self.enc_hidden_size, hidden_size)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embed_size + self.enc_hidden_size,  # Concatenate embedding và context vector
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size + self.enc_hidden_size + embed_size, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, prev_hidden, prev_cell, encoder_outputs, src_mask=None):
        """
        Args:
            tgt (Tensor): Target tensor [batch_size, 1]
            prev_hidden (Tensor): Previous hidden state [num_layers, batch_size, hidden_size]
            prev_cell (Tensor): Previous cell state [num_layers, batch_size, hidden_size]
            encoder_outputs (Tensor): Outputs của encoder [batch_size, src_len, enc_hidden_size]
            src_mask (Tensor, optional): Mask cho source sequence [batch_size, src_len]
            
        Returns:
            tuple: (output, hidden, cell, attention_weights)
                - output: Tensor [batch_size, vocab_size]
                - hidden: Tensor [num_layers, batch_size, hidden_size]
                - cell: Tensor [num_layers, batch_size, hidden_size]
                - attention_weights: Tensor [batch_size, src_len]
        """
        # Embedding
        embedded = self.dropout(self.embedding(tgt))  # [batch_size, 1, embed_size]
        
        # Tính attention
        attention_weights, context_vector = self.attention(prev_hidden[-1], encoder_outputs, src_mask)
        
        # Concatenate embedding và context vector
        lstm_input = torch.cat((embedded, context_vector.unsqueeze(1)), dim=2)  # [batch_size, 1, embed_size + enc_hidden_size]
        
        # LSTM
        output, (hidden, cell) = self.lstm(lstm_input, (prev_hidden, prev_cell))
        
        # Concatenate output, context vector và embedding cho output layer
        output = torch.cat((output.squeeze(1), context_vector, embedded.squeeze(1)), dim=1)
        
        # Output layer
        output = self.output_layer(output)  # [batch_size, vocab_size]
        
        return output, hidden, cell, attention_weights


class LSTMSeq2Seq(nn.Module):
    """
    Mô hình LSTM Sequence-to-Sequence với Attention
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size=256, hidden_size=512, 
                 enc_num_layers=2, dec_num_layers=2, dropout=0.3, bidirectional=True, pad_idx=0):
        """
        Khởi tạo mô hình LSTM Seq2Seq
        
        Args:
            src_vocab_size (int): Kích thước từ điển nguồn
            tgt_vocab_size (int): Kích thước từ điển đích
            embed_size (int): Kích thước embedding
            hidden_size (int): Kích thước hidden state
            enc_num_layers (int): Số lớp LSTM trong encoder
            dec_num_layers (int): Số lớp LSTM trong decoder
            dropout (float): Tỷ lệ dropout
            bidirectional (bool): Có sử dụng LSTM hai chiều trong encoder không
            pad_idx (int): Chỉ số của token padding
        """
        super(LSTMSeq2Seq, self).__init__()
        
        self.pad_idx = pad_idx
        self.hidden_size = hidden_size
        
        # Encoder
        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=enc_num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        # Decoder
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            enc_hidden_size=hidden_size * (2 if bidirectional else 1),
            num_layers=dec_num_layers,
            dropout=dropout
        )
        
        # Criterion
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        
    def forward(self, src, tgt, src_lengths, teacher_forcing_ratio=0.5):
        """
        Args:
            src (Tensor): Source tensor [batch_size, src_len]
            tgt (Tensor): Target tensor [batch_size, tgt_len]
            src_lengths (Tensor): Độ dài thực của các câu nguồn [batch_size]
            teacher_forcing_ratio (float): Tỷ lệ sử dụng teacher forcing
            
        Returns:
            tuple: (outputs, attention_weights)
                - outputs: Tensor [batch_size, tgt_len, vocab_size]
                - attention_weights: Tensor [batch_size, tgt_len, src_len]
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = self.decoder.vocab_size
        
        # Tensor để lưu outputs
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(src.device)
        attention_weights = torch.zeros(batch_size, tgt_len, src.size(1)).to(src.device)
        
        # Tạo mask cho source sequence
        src_mask = (src != self.pad_idx).to(src.device)  # [batch_size, src_len]
        
        # Encoder
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
        
        # Decoder - bắt đầu với token đầu tiên
        input = tgt[:, 0].unsqueeze(1)  # [batch_size, 1]
        
        for t in range(1, tgt_len):
            # Decoder step
            output, hidden, cell, attn_weights = self.decoder(
                input, hidden, cell, encoder_outputs, src_mask
            )
            
            # Lưu output và attention weights
            outputs[:, t, :] = output
            attention_weights[:, t, :] = attn_weights
            
            # Teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            
            # Lấy token tiếp theo (từ ground truth hoặc từ prediction)
            top1 = output.argmax(1)
            input = tgt[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
        
        return outputs, attention_weights
    
    def calculate_loss(self, outputs, target):
        """
        Tính loss
        
        Args:
            outputs (Tensor): Output tensor [batch_size, tgt_len, vocab_size]
            target (Tensor): Target tensor [batch_size, tgt_len]
            
        Returns:
            Tensor: Loss
        """
        # Bỏ qua token đầu tiên (BOS)
        outputs = outputs[:, 1:, :].contiguous().view(-1, outputs.size(2))
        target = target[:, 1:].contiguous().view(-1)
        
        return self.criterion(outputs, target)
    
    def translate(self, src, src_lengths, tgt_sp, max_len=100, device=None):
        """
        Dịch một câu từ ngôn ngữ nguồn sang ngôn ngữ đích
        
        Args:
            src (Tensor): Source tensor [batch_size, src_len]
            src_lengths (Tensor): Độ dài thực của các câu nguồn [batch_size]
            tgt_sp: Tokenizer đích
            max_len (int): Độ dài tối đa của câu dịch
            device: Thiết bị (CPU/GPU)
            
        Returns:
            tuple: (translations, attention_weights)
                - translations: List[str] - Danh sách các câu dịch
                - attention_weights: Tensor [batch_size, max_len, src_len]
        """
        if device is None:
            device = src.device
            
        batch_size = src.size(0)
        
        # Tensor để lưu attention weights
        attention_weights = torch.zeros(batch_size, max_len, src.size(1)).to(device)
        
        # Tạo mask cho source sequence
        src_mask = (src != self.pad_idx).to(device)  # [batch_size, src_len]
        
        # Encoder
        with torch.no_grad():
            encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
        
        # Bắt đầu với token BOS
        input = torch.tensor([[tgt_sp.bos_id()]] * batch_size).to(device)
        
        # Lưu các token đã dịch
        translated_tokens = torch.zeros(batch_size, max_len).long().to(device)
        translated_tokens[:, 0] = tgt_sp.bos_id()
        
        # Dịch từng token
        for t in range(1, max_len):
            # Decoder step
            with torch.no_grad():
                output, hidden, cell, attn_weights = self.decoder(
                    input, hidden, cell, encoder_outputs, src_mask
                )
            
            # Lưu attention weights
            attention_weights[:, t, :] = attn_weights
            
            # Lấy token có xác suất cao nhất
            pred_token = output.argmax(1)
            translated_tokens[:, t] = pred_token
            
            # Cập nhật input cho bước tiếp theo
            input = pred_token.unsqueeze(1)
            
            # Kiểm tra nếu tất cả các câu đã kết thúc
            if (pred_token == tgt_sp.eos_id()).all():
                break
        
        # Chuyển từ tokens sang câu
        translations = []
        for i in range(batch_size):
            tokens = translated_tokens[i, :].tolist()
            
            # Tìm vị trí của token EOS đầu tiên
            if tgt_sp.eos_id() in tokens:
                eos_idx = tokens.index(tgt_sp.eos_id())
                tokens = tokens[1:eos_idx]  # Bỏ token BOS và EOS
            else:
                tokens = tokens[1:]  # Chỉ bỏ token BOS
            
            translations.append(tgt_sp.decode(tokens))
        
        return translations, attention_weights
    
    def count_parameters(self):
        """
        Đếm số lượng tham số của mô hình
        
        Returns:
            int: Số lượng tham số
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_lstm_model(config, src_vocab_size, tgt_vocab_size):
    """
    Tạo mô hình LSTM Seq2Seq
    
    Args:
        config (dict): Cấu hình mô hình
        src_vocab_size (int): Kích thước từ điển nguồn
        tgt_vocab_size (int): Kích thước từ điển đích
        
    Returns:
        LSTMSeq2Seq: Mô hình LSTM Seq2Seq
    """
    model = LSTMSeq2Seq(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        embed_size=config['embed_size'],
        hidden_size=config['hidden_size'],
        enc_num_layers=config['enc_num_layers'],
        dec_num_layers=config['dec_num_layers'],
        dropout=config['dropout'],
        bidirectional=config['bidirectional'],
        pad_idx=config['pad_idx']
    )
    
    logger.info(f"Đã tạo mô hình LSTM Seq2Seq với {model.count_parameters():,} tham số")
    
    return model


if __name__ == "__main__":
    # Cấu hình mặc định
    config = {
        'embed_size': 256,
        'hidden_size': 512,
        'enc_num_layers': 2,
        'dec_num_layers': 2,
        'dropout': 0.3,
        'bidirectional': True,
        'pad_idx': 0
    }
    
    # Tạo mô hình
    model = create_lstm_model(config, src_vocab_size=8000, tgt_vocab_size=8000)
    logger.info(f"Số lượng tham số: {model.count_parameters():,}")
