#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Huấn luyện mô hình dịch máy Tiếng Việt - Tiếng Anh
"""

import os
import time
import math
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import sacrebleu
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Import các module đã tạo
from data_preprocessing import TranslationDataset, collate_fn, create_dataloaders
from transformer_model import TransformerModel, NoamOpt, create_transformer_model
from lstm_model import LSTMSeq2Seq, create_lstm_model

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Lớp quản lý quá trình huấn luyện mô hình
    """
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, 
                 scheduler=None, clip_grad=1.0, device='cuda', model_type='transformer',
                 mixed_precision=False, gradient_accumulation_steps=1):
        """
        Khởi tạo Trainer
        
        Args:
            model: Mô hình cần huấn luyện (Transformer hoặc LSTM)
            train_loader: DataLoader cho tập huấn luyện
            val_loader: DataLoader cho tập validation
            optimizer: Optimizer
            criterion: Loss function
            scheduler: Learning rate scheduler
            clip_grad: Ngưỡng gradient clipping
            device: Thiết bị (CPU/GPU)
            model_type: Loại mô hình ('transformer' hoặc 'lstm')
            mixed_precision: Có sử dụng mixed precision hay không
            gradient_accumulation_steps: Số bước tích lũy gradient
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.clip_grad = clip_grad
        self.device = device
        self.model_type = model_type
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Khởi tạo scaler cho mixed precision
        self.scaler = GradScaler() if mixed_precision else None
        
        # Chuyển mô hình sang thiết bị
        self.model = self.model.to(self.device)
    
    def train_epoch(self):
        """
        Huấn luyện một epoch
        
        Returns:
            float: Loss trung bình
        """
        self.model.train()
        epoch_loss = 0
        
        # Tạo progress bar
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        # Đặt lại gradient
        self.optimizer.zero_grad()
        
        for i, (src, tgt, src_lengths, tgt_lengths) in enumerate(progress_bar):
            # Chuyển dữ liệu sang thiết bị
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            src_lengths = src_lengths.to(self.device)
            
            # Tính toán với mixed precision nếu được bật
            if self.mixed_precision:
                with autocast():
                    # Forward pass
                    if self.model_type == 'transformer':
                        # Transformer model
                        output = self.model(src, tgt[:, :-1])
                        output = output.contiguous().view(-1, output.shape[-1])
                        tgt = tgt[:, 1:].contiguous().view(-1)
                    else:
                        # LSTM model
                        output = self.model(src, src_lengths, tgt)
                        output = output[:, 1:].contiguous().view(-1, output.shape[-1])
                        tgt = tgt[:, 1:].contiguous().view(-1)
                    
                    # Tính loss
                    loss = self.criterion(output, tgt)
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass với gradient accumulation
                self.scaler.scale(loss).backward()
                
                if (i + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                    # Scheduler step nếu có
                    if self.scheduler is not None:
                        self.scheduler.step()
                    
                    # Đặt lại gradient
                    self.optimizer.zero_grad()
            else:
                # Forward pass
                if self.model_type == 'transformer':
                    # Transformer model
                    output = self.model(src, tgt[:, :-1])
                    output = output.contiguous().view(-1, output.shape[-1])
                    tgt = tgt[:, 1:].contiguous().view(-1)
                else:
                    # LSTM model
                    output = self.model(src, src_lengths, tgt)
                    output = output[:, 1:].contiguous().view(-1, output.shape[-1])
                    tgt = tgt[:, 1:].contiguous().view(-1)
                
                # Tính loss
                loss = self.criterion(output, tgt)
                loss = loss / self.gradient_accumulation_steps
                
                # Backward pass với gradient accumulation
                loss.backward()
                
                if (i + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                    
                    # Optimizer step
                    self.optimizer.step()
                    
                    # Scheduler step nếu có
                    if self.scheduler is not None:
                        self.scheduler.step()
                    
                    # Đặt lại gradient
                    self.optimizer.zero_grad()
            
            # Cập nhật loss
            epoch_loss += loss.item() * self.gradient_accumulation_steps
            
            # Cập nhật progress bar
            progress_bar.set_postfix({"loss": f"{epoch_loss / (i + 1):.4f}"})
        
        return epoch_loss / len(self.train_loader)
    
    def evaluate(self):
        """
        Đánh giá mô hình trên tập validation
        
        Returns:
            float: Loss trung bình
        """
        self.model.eval()
        epoch_loss = 0
        
        with torch.no_grad():
            for src, tgt, src_lengths, tgt_lengths in self.val_loader:
                # Chuyển dữ liệu sang thiết bị
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                src_lengths = src_lengths.to(self.device)
                
                # Forward pass
                if self.model_type == 'transformer':
                    # Transformer model
                    output = self.model(src, tgt[:, :-1])
                    output = output.contiguous().view(-1, output.shape[-1])
                    tgt = tgt[:, 1:].contiguous().view(-1)
                else:
                    # LSTM model
                    output = self.model(src, src_lengths, tgt, teacher_forcing_ratio=0.0)
                    output = output[:, 1:].contiguous().view(-1, output.shape[-1])
                    tgt = tgt[:, 1:].contiguous().view(-1)
                
                # Tính loss
                loss = self.criterion(output, tgt)
                
                # Cập nhật loss
                epoch_loss += loss.item()
        
        return epoch_loss / len(self.val_loader)
    
    def train(self, epochs, model_dir, patience=5):
        """
        Huấn luyện mô hình
        
        Args:
            epochs (int): Số epoch
            model_dir (str): Thư mục lưu mô hình
            patience (int): Số epoch chờ đợi trước khi early stopping
        
        Returns:
            dict: Lịch sử huấn luyện
        """
        # Tạo thư mục lưu mô hình
        os.makedirs(model_dir, exist_ok=True)
        
        # Khởi tạo biến theo dõi
        best_val_loss = float('inf')
        patience_counter = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
        # Bắt đầu huấn luyện
        logger.info(f"Bắt đầu huấn luyện mô hình {self.model_type}...")
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            # Huấn luyện một epoch
            train_loss = self.train_epoch()
            
            # Đánh giá mô hình
            val_loss = self.evaluate()
            
            # Lưu lịch sử
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Lưu learning rate
            if self.scheduler is not None:
                if hasattr(self.scheduler, '_rate'):
                    # NoamOpt scheduler
                    lr = self.scheduler._rate
                else:
                    # Scheduler thông thường
                    lr = self.scheduler.get_last_lr()[0]
            else:
                lr = self.optimizer.param_groups[0]['lr']
            
            history['learning_rates'].append(lr)
            
            # Tính thời gian
            end_time = time.time()
            epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
            
            # In thông tin
            logger.info(f"Epoch: {epoch:02} | Time: {epoch_mins}m {epoch_secs:.0f}s")
            logger.info(f"\tTrain Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Lưu mô hình tốt nhất
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Lưu mô hình
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, os.path.join(model_dir, f"{self.model_type}_best.pt"))
                
                logger.info(f"\tMô hình tốt nhất đã được lưu!")
            else:
                patience_counter += 1
                logger.info(f"\tPatience: {patience_counter}/{patience}")
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping sau {epoch} epochs!")
                break
            
            # Lưu mô hình cuối cùng
            if epoch == epochs:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, os.path.join(model_dir, f"{self.model_type}_final.pt"))
        
        # Vẽ biểu đồ
        self._plot_history(history, model_dir)
        
        return history
    
    def _plot_history(self, history, model_dir):
        """
        Vẽ biểu đồ lịch sử huấn luyện
        
        Args:
            history (dict): Lịch sử huấn luyện
            model_dir (str): Thư mục lưu biểu đồ
        """
        # Tạo figure
        plt.figure(figsize=(12, 8))
        
        # Vẽ loss
        plt.subplot(2, 1, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title(f'{self.model_type.capitalize()} Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Vẽ learning rate
        plt.subplot(2, 1, 2)
        plt.plot(history['learning_rates'])
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        
        # Lưu biểu đồ
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, f"{self.model_type}_history.png"))
        plt.close()


def train_transformer(config):
    """
    Huấn luyện mô hình Transformer
    
    Args:
        config (dict): Cấu hình
    
    Returns:
        tuple: (model, history) - Mô hình và lịch sử huấn luyện
    """
    # Tải tokenizer
    with open(os.path.join(config['model_dir'], 'tokenizers.pkl'), 'rb') as f:
        tokenizers = pickle.load(f)
    
    en_sp = tokenizers['en_sp']
    vi_sp = tokenizers['vi_sp']
    
    # Tải dữ liệu
    train_data = pd.read_csv(os.path.join(config['data_dir'], 'train_data.csv'))
    val_data = pd.read_csv(os.path.join(config['data_dir'], 'val_data.csv'))
    test_data = pd.read_csv(os.path.join(config['data_dir'], 'test_data.csv'))
    
    # Tạo DataLoader
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data, en_sp, vi_sp, config
    )
    
    # Xác định kích thước từ điển
    src_vocab_size = en_sp.get_piece_size() if config['src_lang'] == 'en' else vi_sp.get_piece_size()
    tgt_vocab_size = vi_sp.get_piece_size() if config['tgt_lang'] == 'vi' else en_sp.get_piece_size()
    
    # Tạo mô hình
    model = create_transformer_model(config['transformer'], src_vocab_size, tgt_vocab_size)
    
    # Tạo optimizer và scheduler
    optimizer = optim.Adam(model.parameters(), lr=config['transformer']['learning_rate'], betas=(0.9, 0.98), eps=1e-9)
    
    # Sử dụng NoamOpt scheduler
    scheduler = NoamOpt(
        model_size=config['transformer']['d_model'],
        factor=config['transformer']['lr_factor'],
        warmup=config['transformer']['warmup_steps'],
        optimizer=optimizer
    )
    
    # Tạo loss function với label smoothing
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=config['transformer']['label_smoothing'])
    
    # Tạo trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=scheduler,  # NoamOpt đã bao gồm optimizer
        criterion=criterion,
        scheduler=None,  # Không cần scheduler vì NoamOpt đã xử lý
        clip_grad=config['transformer']['clip_grad'],
        device=config['device'],
        model_type='transformer',
        mixed_precision=config['mixed_precision'],
        gradient_accumulation_steps=config['gradient_accumulation_steps']
    )
    
    # Huấn luyện mô hình
    history = trainer.train(
        epochs=config['transformer']['epochs'],
        model_dir=config['model_dir'],
        patience=config['transformer']['patience']
    )
    
    return model, history


def train_lstm(config):
    """
    Huấn luyện mô hình LSTM
    
    Args:
        config (dict): Cấu hình
    
    Returns:
        tuple: (model, history) - Mô hình và lịch sử huấn luyện
    """
    # Tải tokenizer
    with open(os.path.join(config['model_dir'], 'tokenizers.pkl'), 'rb') as f:
        tokenizers = pickle.load(f)
    
    en_sp = tokenizers['en_sp']
    vi_sp = tokenizers['vi_sp']
    
    # Tải dữ liệu
    train_data = pd.read_csv(os.path.join(config['data_dir'], 'train_data.csv'))
    val_data = pd.read_csv(os.path.join(config['data_dir'], 'val_data.csv'))
    test_data = pd.read_csv(os.path.join(config['data_dir'], 'test_data.csv'))
    
    # Tạo DataLoader
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data, en_sp, vi_sp, config
    )
    
    # Xác định kích thước từ điển
    src_vocab_size = en_sp.get_piece_size() if config['src_lang'] == 'en' else vi_sp.get_piece_size()
    tgt_vocab_size = vi_sp.get_piece_size() if config['tgt_lang'] == 'vi' else en_sp.get_piece_size()
    
    # Tạo mô hình
    model = create_lstm_model(config['lstm'], src_vocab_size, tgt_vocab_size)
    
    # Tạo optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['lstm']['learning_rate'])
    
    # Tạo scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Tạo loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Tạo trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        clip_grad=config['lstm']['clip_grad'],
        device=config['device'],
        model_type='lstm',
        mixed_precision=config['mixed_precision'],
        gradient_accumulation_steps=config['gradient_accumulation_steps']
    )
    
    # Huấn luyện mô hình
    history = trainer.train(
        epochs=config['lstm']['epochs'],
        model_dir=config['model_dir'],
        patience=config['lstm']['patience']
    )
    
    return model, history


def compare_models(transformer_history, lstm_history, config):
    """
    So sánh hiệu suất của hai mô hình
    
    Args:
        transformer_history (dict): Lịch sử huấn luyện của Transformer
        lstm_history (dict): Lịch sử huấn luyện của LSTM
        config (dict): Cấu hình
    """
    # Tạo figure
    plt.figure(figsize=(12, 6))
    
    # Vẽ train loss
    plt.subplot(1, 2, 1)
    plt.plot(transformer_history['train_loss'], label='Transformer')
    plt.plot(lstm_history['train_loss'], label='LSTM')
    plt.title('Train Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Vẽ validation loss
    plt.subplot(1, 2, 2)
    plt.plot(transformer_history['val_loss'], label='Transformer')
    plt.plot(lstm_history['val_loss'], label='LSTM')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Lưu biểu đồ
    plt.tight_layout()
    plt.savefig(os.path.join(config['model_dir'], "model_comparison.png"))
    plt.close()
    
    # Tạo DataFrame so sánh
    comparison = pd.DataFrame({
        'Transformer Train Loss': transformer_history['train_loss'],
        'Transformer Val Loss': transformer_history['val_loss'],
        'LSTM Train Loss': lstm_history['train_loss'],
        'LSTM Val Loss': lstm_history['val_loss']
    })
    
    # Lưu DataFrame
    comparison.to_csv(os.path.join(config['model_dir'], "model_comparison.csv"), index=False)
    
    # In kết quả
    logger.info("\nSo sánh hiệu suất của hai mô hình:")
    logger.info(f"Transformer - Val Loss cuối cùng: {transformer_history['val_loss'][-1]:.4f}")
    logger.info(f"LSTM - Val Loss cuối cùng: {lstm_history['val_loss'][-1]:.4f}")
    
    # Xác định mô hình tốt hơn
    if min(transformer_history['val_loss']) < min(lstm_history['val_loss']):
        logger.info("Kết luận: Mô hình Transformer có hiệu suất tốt hơn!")
    else:
        logger.info("Kết luận: Mô hình LSTM có hiệu suất tốt hơn!")


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
        'mixed_precision': True,
        'gradient_accumulation_steps': 4,
        
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
    
    # Huấn luyện mô hình Transformer
    transformer_model, transformer_history = train_transformer(config)
    
    # Huấn luyện mô hình LSTM
    lstm_model, lstm_history = train_lstm(config)
    
    # So sánh hai mô hình
    compare_models(transformer_history, lstm_history, config)
