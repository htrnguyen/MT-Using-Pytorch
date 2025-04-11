#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mô hình dịch máy LSTM cho Tiếng Việt - Tiếng Anh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    """
    Encoder của mô hình LSTM Seq2Seq
    """
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout, bidirectional, pad_idx):
        """
        Khởi tạo Encoder
        
        Args:
            vocab_size (int): Kích thước từ điển
            embed_size (int): Kích thước embedding
            hidden_size (int): Kích thước hidden state
            num_layers (int): Số lượng lớp LSTM
            dropout (float): Tỷ lệ dropout
            bidirectional (bool): Có sử dụng LSTM hai chiều hay không
            pad_idx (int): Chỉ số của token padding
        """
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout_rate = dropout
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_lengths):
        """
        Args:
            src: Tensor shape [batch_size, src_len]
            src_lengths: Tensor shape [batch_size]
        
        Returns:
            outputs: Tensor shape [batch_size, src_len, hidden_size * num_directions]
            hidden: Tuple of (h_n, c_n) với shape [num_layers * num_directions, batch_size, hidden_size]
        """
        # Embedding
        embedded = self.dropout(self.embedding(src))
        
        # Pack padded sequence
        packed = pack_padded_sequence(embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # LSTM
        outputs, hidden = self.lstm(packed)
        
        # Unpack sequence
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        
        # Xử lý hidden state nếu là LSTM hai chiều
        if self.bidirectional:
            # Kết hợp hidden state từ hai chiều
            h_n, c_n = hidden
            h_n = self._combine_bidirectional(h_n)
            c_n = self._combine_bidirectional(c_n)
            hidden = (h_n, c_n)
        
        return outputs, hidden
    
    def _combine_bidirectional(self, hidden):
        """
        Kết hợp hidden state từ hai chiều
        
        Args:
            hidden: Tensor shape [num_layers * num_directions, batch_size, hidden_size]
        
        Returns:
            Tensor shape [num_layers, batch_size, hidden_size * num_directions]
        """
        num_layers = self.num_layers
        batch_size = hidden.shape[1]
        hidden_size = self.hidden_size
        
        # Tách hidden state theo chiều
        hidden = hidden.view(num_layers, 2, batch_size, hidden_size)
        
        # Kết hợp hidden state từ hai chiều
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)
        
        return hidden


class Attention(nn.Module):
    """
    Cơ chế Attention cho Decoder
    """
    def __init__(self, enc_hidden_size, dec_hidden_size, bidirectional=True):
        """
        Khởi tạo Attention
        
        Args:
            enc_hidden_size (int): Kích thước hidden state của Encoder
            dec_hidden_size (int): Kích thước hidden state của Decoder
            bidirectional (bool): Có sử dụng LSTM hai chiều hay không
        """
        super(Attention, self).__init__()
        
        # Kích thước hidden state của Encoder sau khi kết hợp hai chiều
        self.enc_hidden_size = enc_hidden_size * 2 if bidirectional else enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        
        # Layer tính attention score
        self.attn = nn.Linear(self.enc_hidden_size + dec_hidden_size, dec_hidden_size)
        self.v = nn.Linear(dec_hidden_size, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs, mask=None):
        """
        Args:
            hidden: Tensor shape [batch_size, dec_hidden_size]
            encoder_outputs: Tensor shape [batch_size, src_len, enc_hidden_size]
            mask: Tensor shape [batch_size, src_len]
        
        Returns:
            attention: Tensor shape [batch_size, src_len]
            context: Tensor shape [batch_size, enc_hidden_size]
        """
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Lặp lại hidden state để kết hợp với từng encoder output
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # Kết hợp hidden state và encoder outputs
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        
        # Tính attention score
        attention = self.v(energy).squeeze(2)
        
        # Áp dụng mask nếu có
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
        
        # Áp dụng softmax để có attention weights
        attention = F.softmax(attention, dim=1)
        
        # Tính context vector
        context = torch.bmm(attention.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return attention, context


class Decoder(nn.Module):
    """
    Decoder của mô hình LSTM Seq2Seq với Attention
    """
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, 
                 num_layers, dropout, bidirectional_encoder, pad_idx):
        """
        Khởi tạo Decoder
        
        Args:
            vocab_size (int): Kích thước từ điển
            embed_size (int): Kích thước embedding
            enc_hidden_size (int): Kích thước hidden state của Encoder
            dec_hidden_size (int): Kích thước hidden state của Decoder
            num_layers (int): Số lượng lớp LSTM
            dropout (float): Tỷ lệ dropout
            bidirectional_encoder (bool): Encoder có sử dụng LSTM hai chiều hay không
            pad_idx (int): Chỉ số của token padding
        """
        super(Decoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.dec_hidden_size = dec_hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        # Kích thước hidden state của Encoder sau khi kết hợp hai chiều
        self.enc_hidden_size = enc_hidden_size * 2 if bidirectional_encoder else enc_hidden_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        
        # Attention
        self.attention = Attention(enc_hidden_size, dec_hidden_size, bidirectional_encoder)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embed_size + self.enc_hidden_size,
            hidden_size=dec_hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.fc_out = nn.Linear(dec_hidden_size + self.enc_hidden_size + embed_size, vocab_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs, mask=None):
        """
        Args:
            input: Tensor shape [batch_size, 1]
            hidden: Tuple of (h_n, c_n) với shape [num_layers, batch_size, dec_hidden_size]
            encoder_outputs: Tensor shape [batch_size, src_len, enc_hidden_size]
            mask: Tensor shape [batch_size, src_len]
        
        Returns:
            output: Tensor shape [batch_size, vocab_size]
            hidden: Tuple of (h_n, c_n) với shape [num_layers, batch_size, dec_hidden_size]
            attention: Tensor shape [batch_size, src_len]
        """
        # Embedding
        embedded = self.dropout(self.embedding(input))
        
        # Tính attention và context vector
        h_n, c_n = hidden
        attention, context = self.attention(h_n[-1], encoder_outputs, mask)
        
        # Kết hợp embedded và context vector
        lstm_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)
        
        # LSTM
        output, hidden = self.lstm(lstm_input, hidden)
        
        # Kết hợp output, context và embedded để dự đoán
        output = output.squeeze(1)
        embedded = embedded.squeeze(1)
        context = context
        
        output = self.fc_out(torch.cat((output, context, embedded), dim=1))
        
        return output, hidden, attention


class LSTMSeq2Seq(nn.Module):
    """
    Mô hình LSTM Seq2Seq đầy đủ với Attention
    """
    def __init__(self, encoder, decoder, device):
        """
        Khởi tạo mô hình LSTM Seq2Seq
        
        Args:
            encoder (Encoder): Encoder
            decoder (Decoder): Decoder
            device (str): Thiết bị (CPU/GPU)
        """
        super(LSTMSeq2Seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, src_lengths, tgt, teacher_forcing_ratio=0.5):
        """
        Args:
            src: Tensor shape [batch_size, src_len]
            src_lengths: Tensor shape [batch_size]
            tgt: Tensor shape [batch_size, tgt_len]
            teacher_forcing_ratio (float): Tỷ lệ sử dụng teacher forcing
        
        Returns:
            outputs: Tensor shape [batch_size, tgt_len, vocab_size]
        """
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.vocab_size
        
        # Tensor để lưu kết quả
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        
        # Encoder
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        
        # Tạo mask cho source padding
        mask = (src != 0).to(self.device)
        
        # Bắt đầu với token BOS
        input = tgt[:, 0].unsqueeze(1)
        
        for t in range(1, tgt_len):
            # Decoder
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)
            
            # Lưu kết quả
            outputs[:, t, :] = output
            
            # Quyết định sử dụng teacher forcing hay không
            teacher_force = random.random() < teacher_forcing_ratio
            
            # Lấy token có xác suất cao nhất
            top1 = output.argmax(1)
            
            # Chuẩn bị input cho bước tiếp theo
            input = tgt[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
        
        return outputs
    
    def translate(self, src, src_lengths, tgt_sp, max_len=100, device='cuda'):
        """
        Dịch một câu
        
        Args:
            src: Tensor shape [batch_size, src_len]
            src_lengths: Tensor shape [batch_size]
            tgt_sp: Tokenizer đích
            max_len (int): Độ dài tối đa của câu dịch
            device (str): Thiết bị (CPU/GPU)
        
        Returns:
            translations (list): Danh sách các câu đã dịch
            attention_weights (list): Danh sách các attention weights
        """
        batch_size = src.shape[0]
        
        # Encoder
        with torch.no_grad():
            encoder_outputs, hidden = self.encoder(src, src_lengths)
        
        # Tạo mask cho source padding
        mask = (src != 0).to(device)
        
        # Bắt đầu với token BOS
        input = torch.tensor([tgt_sp.bos_id()] * batch_size).unsqueeze(1).to(device)
        
        translations = [''] * batch_size
        attention_weights = [[] for _ in range(batch_size)]
        
        for t in range(max_len):
            with torch.no_grad():
                output, hidden, attention = self.decoder(input, hidden, encoder_outputs, mask)
            
            # Lấy token có xác suất cao nhất
            pred_token = output.argmax(1)
            
            # Lưu attention weights
            for i in range(batch_size):
                attention_weights[i].append(attention[i].cpu().numpy())
            
            # Cập nhật translations
            for i in range(batch_size):
                token = pred_token[i].item()
                if token == tgt_sp.eos_id():
                    continue
                if translations[i] == '':
                    translations[i] = tgt_sp.id_to_piece(token)
                else:
                    translations[i] += tgt_sp.id_to_piece(token).replace('▁', ' ')
            
            # Kiểm tra nếu tất cả các câu đều kết thúc
            if all(pred_token.eq(tgt_sp.eos_id())):
                break
            
            # Chuẩn bị input cho bước tiếp theo
            input = pred_token.unsqueeze(1)
        
        # Xử lý kết quả
        for i in range(batch_size):
            translations[i] = translations[i].replace(' ', '')
        
        return translations, attention_weights


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
    # Tạo Encoder
    encoder = Encoder(
        vocab_size=src_vocab_size,
        embed_size=config['embed_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['enc_num_layers'],
        dropout=config['dropout'],
        bidirectional=config['bidirectional'],
        pad_idx=config['pad_idx']
    )
    
    # Tạo Decoder
    decoder = Decoder(
        vocab_size=tgt_vocab_size,
        embed_size=config['embed_size'],
        enc_hidden_size=config['hidden_size'],
        dec_hidden_size=config['hidden_size'],
        num_layers=config['dec_num_layers'],
        dropout=config['dropout'],
        bidirectional_encoder=config['bidirectional'],
        pad_idx=config['pad_idx']
    )
    
    # Tạo mô hình
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMSeq2Seq(encoder, decoder, device)
    
    # Khởi tạo trọng số
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return model
