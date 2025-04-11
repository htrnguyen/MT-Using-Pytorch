#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Huấn luyện và tối ưu hóa mô hình dịch máy Tiếng Việt - Tiếng Anh
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
from data_preprocessing import TranslationDataset, collate_fn
from transformer_model import TransformerModel, NoamOpt
from lstm_model import LSTMSeq2Seq

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
                 mixed_precision=True, gradient_accumulation_steps=1):
        """
        Khởi tạo Trainer
        
        Args:
            model: Mô hình cần huấn luyện (Transformer hoặc LSTM)
            train_loader: DataLoader cho tập huấn luyện
            val_loader: DataLoader cho tập validation
            optimizer: Optimizer
            criterion: Loss function
            scheduler: Learning rate scheduler
            clip_grad (float): Ngưỡng clip gradient
            device (str): Thiết bị (CPU/GPU)
            model_type (str): Loại mô hình ('transformer' hoặc 'lstm')
            mixed_precision (bool): Có sử dụng mixed precision không
            gradient_accumulation_steps (int): Số bước tích lũy gradient
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
        
        # Lưu lịch sử huấn luyện
        self.train_losses = []
        self.val_losses = []
        self.train_ppls = []
        self.val_ppls = []
        self.learning_rates = []
        
        # Lưu thời gian huấn luyện
        self.train_times = []
        
        # Lưu thông tin về early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self):
        """
        Huấn luyện một epoch
        
        Returns:
            float: Loss trung bình
        """
        self.model.train()
        epoch_loss = 0
        batch_count = 0
        start_time = time.time()
        
        # Sử dụng tqdm để hiển thị tiến độ
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for i, batch in enumerate(progress_bar):
            # Xử lý batch tùy theo loại mô hình
            if self.model_type == 'transformer':
                src, tgt, src_mask, tgt_mask = batch
                src, tgt = src.to(self.device), tgt.to(self.device)
                src_mask, tgt_mask = src_mask.to(self.device), tgt_mask.to(self.device)
                
                # Chuẩn bị input và target
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                # Forward pass với mixed precision
                if self.mixed_precision:
                    with autocast():
                        output = self.model(src, tgt_input, src_mask, tgt_mask[:, :, :-1, :-1])
                        loss = self.model.calculate_loss(output, tgt_output)
                        loss = loss / self.gradient_accumulation_steps  # Normalize loss
                else:
                    output = self.model(src, tgt_input, src_mask, tgt_mask[:, :, :-1, :-1])
                    loss = self.model.calculate_loss(output, tgt_output)
                    loss = loss / self.gradient_accumulation_steps  # Normalize loss
                
            elif self.model_type == 'lstm':
                src, tgt, src_mask, _ = batch
                src, tgt = src.to(self.device), tgt.to(self.device)
                
                # Tính độ dài thực của các câu nguồn
                src_lengths = src_mask.sum(dim=1).int()
                
                # Forward pass với mixed precision
                if self.mixed_precision:
                    with autocast():
                        outputs, _ = self.model(src, tgt, src_lengths)
                        loss = self.model.calculate_loss(outputs, tgt)
                        loss = loss / self.gradient_accumulation_steps  # Normalize loss
                else:
                    outputs, _ = self.model(src, tgt, src_lengths)
                    loss = self.model.calculate_loss(outputs, tgt)
                    loss = loss / self.gradient_accumulation_steps  # Normalize loss
            
            # Backward pass với mixed precision
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Tích lũy gradient
            if (i + 1) % self.gradient_accumulation_steps == 0 or (i + 1) == len(self.train_loader):
                # Clip gradient
                if self.mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                    self.optimizer.step()
                
                # Scheduler step (nếu có)
                if self.scheduler is not None:
                    if isinstance(self.scheduler, NoamOpt):
                        self.scheduler.step()
                    else:
                        self.scheduler.step()
                
                # Zero grad
                self.optimizer.zero_grad()
            
            # Cập nhật loss
            epoch_loss += loss.item() * self.gradient_accumulation_steps
            batch_count += 1
            
            # Cập nhật progress bar
            progress_bar.set_postfix({
                'loss': f"{epoch_loss / batch_count:.4f}",
                'ppl': f"{math.exp(epoch_loss / batch_count):.2f}",
                'lr': f"{self.get_lr():.7f}"
            })
        
        # Tính thời gian huấn luyện
        train_time = time.time() - start_time
        self.train_times.append(train_time)
        
        # Tính loss và perplexity trung bình
        avg_loss = epoch_loss / batch_count
        avg_ppl = math.exp(avg_loss)
        
        # Lưu lịch sử
        self.train_losses.append(avg_loss)
        self.train_ppls.append(avg_ppl)
        self.learning_rates.append(self.get_lr())
        
        return avg_loss, avg_ppl, train_time
    
    def evaluate(self):
        """
        Đánh giá mô hình trên tập validation
        
        Returns:
            float: Loss trung bình
        """
        self.model.eval()
        epoch_loss = 0
        batch_count = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Xử lý batch tùy theo loại mô hình
                if self.model_type == 'transformer':
                    src, tgt, src_mask, tgt_mask = batch
                    src, tgt = src.to(self.device), tgt.to(self.device)
                    src_mask, tgt_mask = src_mask.to(self.device), tgt_mask.to(self.device)
                    
                    # Chuẩn bị input và target
                    tgt_input = tgt[:, :-1]
                    tgt_output = tgt[:, 1:]
                    
                    # Forward pass
                    output = self.model(src, tgt_input, src_mask, tgt_mask[:, :, :-1, :-1])
                    loss = self.model.calculate_loss(output, tgt_output)
                    
                elif self.model_type == 'lstm':
                    src, tgt, src_mask, _ = batch
                    src, tgt = src.to(self.device), tgt.to(self.device)
                    
                    # Tính độ dài thực của các câu nguồn
                    src_lengths = src_mask.sum(dim=1).int()
                    
                    # Forward pass
                    outputs, _ = self.model(src, tgt, src_lengths, teacher_forcing_ratio=0.0)
                    loss = self.model.calculate_loss(outputs, tgt)
                
                # Cập nhật loss
                epoch_loss += loss.item()
                batch_count += 1
        
        # Tính loss và perplexity trung bình
        avg_loss = epoch_loss / batch_count
        avg_ppl = math.exp(avg_loss)
        
        # Lưu lịch sử
        self.val_losses.append(avg_loss)
        self.val_ppls.append(avg_ppl)
        
        return avg_loss, avg_ppl
    
    def train(self, epochs, patience=5, save_dir='./models', save_prefix='model'):
        """
        Huấn luyện mô hình
        
        Args:
            epochs (int): Số epoch
            patience (int): Số epoch chờ đợi trước khi early stopping
            save_dir (str): Thư mục lưu mô hình
            save_prefix (str): Tiền tố cho tên file mô hình
            
        Returns:
            dict: Lịch sử huấn luyện
        """
        logger.info(f"Bắt đầu huấn luyện mô hình {self.model_type} trong {epochs} epochs")
        
        # Tạo thư mục lưu mô hình
        os.makedirs(save_dir, exist_ok=True)
        
        # Lưu mô hình tốt nhất
        best_model_path = os.path.join(save_dir, f"{save_prefix}_best.pt")
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            
            # Huấn luyện một epoch
            train_loss, train_ppl, train_time = self.train_epoch()
            logger.info(f"Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f} | "
                       f"Time: {train_time:.2f}s | LR: {self.get_lr():.7f}")
            
            # Đánh giá trên tập validation
            val_loss, val_ppl = self.evaluate()
            logger.info(f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")
            
            # Lưu mô hình tốt nhất
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # Lưu mô hình
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_ppl': train_ppl,
                    'val_ppl': val_ppl,
                }, best_model_path)
                
                logger.info(f"Đã lưu mô hình tốt nhất tại {best_model_path}")
            else:
                self.patience_counter += 1
                logger.info(f"Validation loss không cải thiện. Patience: {self.patience_counter}/{patience}")
                
                if self.patience_counter >= patience:
                    logger.info("Early stopping!")
                    break
            
            # Lưu mô hình checkpoint
            checkpoint_path = os.path.join(save_dir, f"{save_prefix}_epoch{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_ppl': train_ppl,
                'val_ppl': val_ppl,
            }, checkpoint_path)
        
        # Trả về lịch sử huấn luyện
        history = {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'train_ppl': self.train_ppls,
            'val_ppl': self.val_ppls,
            'lr': self.learning_rates,
            'train_time': self.train_times
        }
        
        return history
    
    def get_lr(self):
        """
        Lấy learning rate hiện tại
        
        Returns:
            float: Learning rate
        """
        if isinstance(self.scheduler, NoamOpt):
            return self.scheduler._rate
        
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        
        return 0


def plot_training_history(history, save_path=None):
    """
    Vẽ biểu đồ lịch sử huấn luyện
    
    Args:
        history (dict): Lịch sử huấn luyện
        save_path (str, optional): Đường dẫn để lưu biểu đồ
    """
    # Tạo figure với 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Validation Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot perplexity
    ax2.plot(history['train_ppl'], label='Train Perplexity', marker='o')
    ax2.plot(history['val_ppl'], label='Validation Perplexity', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perplexity')
    ax2.set_title('Training and Validation Perplexity')
    ax2.legend()
    ax2.grid(True)
    
    # Plot learning rate
    ax3.plot(history['lr'], label='Learning Rate', marker='o')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.grid(True)
    
    # Điều chỉnh layout
    plt.tight_layout()
    
    # Lưu biểu đồ nếu có đường dẫn
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Đã lưu biểu đồ lịch sử huấn luyện tại {save_path}")
    else:
        plt.show()


def calculate_bleu(model, test_loader, src_sp, tgt_sp, device, model_type='transformer', max_len=100):
    """
    Tính BLEU score trên tập test
    
    Args:
        model: Mô hình đã huấn luyện
        test_loader: DataLoader cho tập test
        src_sp, tgt_sp: Tokenizer cho ngôn ngữ nguồn và đích
        device: Thiết bị (CPU/GPU)
        model_type (str): Loại mô hình ('transformer' hoặc 'lstm')
        max_len (int): Độ dài tối đa của câu dịch
        
    Returns:
        float: BLEU score
    """
    model.eval()
    hypotheses = []
    references = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Calculating BLEU"):
            if model_type == 'transformer':
                src, tgt, src_mask, _ = batch
                src, tgt = src.to(device), tgt.to(device)
                src_mask = src_mask.to(device)
                
                # Dịch từng câu trong batch
                for i in range(src.size(0)):
                    src_sentence = src[i].unsqueeze(0)
                    src_mask_i = src_mask[i].unsqueeze(0)
                    
                    # Encoder
                    enc_output = model.encoder(src_sentence, src_mask_i)
                    
                    # Bắt đầu với token BOS
                    tgt_tokens = [tgt_sp.bos_id()]
                    tgt_tensor = torch.tensor([tgt_tokens]).to(device)
                    
                    # Dịch từng token
                    for _ in range(max_len):
                        tgt_mask = model.make_tgt_mask(tgt_tensor)
                        output = model.decoder(tgt_tensor, enc_output, tgt_mask, src_mask_i)
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
                    
                    hyp = tgt_sp.decode(pred_tokens)
                    
                    # Lấy câu tham chiếu
                    ref_tokens = tgt[i].tolist()
                    ref_tokens = [t for t in ref_tokens if t != 0]  # Bỏ padding
                    if ref_tokens[0] == tgt_sp.bos_id():
                        ref_tokens = ref_tokens[1:]  # Bỏ token BOS
                    if ref_tokens[-1] == tgt_sp.eos_id():
                        ref_tokens = ref_tokens[:-1]  # Bỏ token EOS
                    
                    ref = tgt_sp.decode(ref_tokens)
                    
                    hypotheses.append(hyp)
                    references.append([ref])
            
            elif model_type == 'lstm':
                src, tgt, src_mask, _ = batch
                src, tgt = src.to(device), tgt.to(device)
                
                # Tính độ dài thực của các câu nguồn
                src_lengths = src_mask.sum(dim=1).int()
                
                # Dịch các câu
                translations, _ = model.translate(src, src_lengths, tgt_sp, max_len, device)
                
                # Lấy câu tham chiếu
                for i in range(src.size(0)):
                    ref_tokens = tgt[i].tolist()
                    ref_tokens = [t for t in ref_tokens if t != 0]  # Bỏ padding
                    if ref_tokens[0] == tgt_sp.bos_id():
                        ref_tokens = ref_tokens[1:]  # Bỏ token BOS
                    if ref_tokens[-1] == tgt_sp.eos_id():
                        ref_tokens = ref_tokens[:-1]  # Bỏ token EOS
                    
                    ref = tgt_sp.decode(ref_tokens)
                    
                    hypotheses.append(translations[i])
                    references.append([ref])
    
    # Tính BLEU score
    bleu = sacrebleu.corpus_bleu(hypotheses, references)
    
    return bleu.score


def create_dataloaders(config):
    """
    Tạo DataLoader cho tập train, validation và test
    
    Args:
        config (dict): Cấu hình
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, src_sp, tgt_sp)
    """
    # Đọc dữ liệu đã xử lý
    train_data = pd.read_csv(os.path.join(config['data_dir'], 'train_data.csv'))
    val_data = pd.read_csv(os.path.join(config['data_dir'], 'val_data.csv'))
    test_data = pd.read_csv(os.path.join(config['data_dir'], 'test_data.csv'))
    
    # Đọc tokenizer
    with open(os.path.join(config['model_dir'], 'tokenizers.pkl'), 'rb') as f:
        tokenizers = pickle.load(f)
    
    en_sp = tokenizers['en_sp']
    vi_sp = tokenizers['vi_sp']
    
    # Xác định tokenizer cho ngôn ngữ nguồn và đích
    src_sp = en_sp if config['src_lang'] == 'en' else vi_sp
    tgt_sp = vi_sp if config['tgt_lang'] == 'vi' else en_sp
    
    # Tạo dataset
    from data_preprocessing import TranslationDataset
    
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
    
    return train_loader, val_loader, test_loader, src_sp, tgt_sp


def train_transformer(config):
    """
    Huấn luyện mô hình Transformer
    
    Args:
        config (dict): Cấu hình
        
    Returns:
        tuple: (model, history)
    """
    # Tạo DataLoader
    train_loader, val_loader, test_loader, src_sp, tgt_sp = create_dataloaders(config)
    
    # Xác định kích thước từ điển
    src_vocab_size = src_sp.get_piece_size()
    tgt_vocab_size = tgt_sp.get_piece_size()
    
    # Tạo mô hình
    from transformer_model import create_transformer_model, NoamOpt
    
    model = create_transformer_model(config['transformer'], src_vocab_size, tgt_vocab_size)
    model = model.to(config['device'])
    
    # Tạo optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['transformer']['learning_rate'],
        betas=(0.9, 0.98), 
        eps=1e-9
    )
    
    # Tạo scheduler
    scheduler = NoamOpt(
        model_size=config['transformer']['d_model'],
        factor=config['transformer']['lr_factor'],
        warmup=config['transformer']['warmup_steps'],
        optimizer=optimizer
    )
    
    # Tạo trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=None,  # Sử dụng criterion trong mô hình
        scheduler=scheduler,
        clip_grad=config['transformer']['clip_grad'],
        device=config['device'],
        model_type='transformer',
        mixed_precision=config['mixed_precision'],
        gradient_accumulation_steps=config['gradient_accumulation_steps']
    )
    
    # Huấn luyện mô hình
    history = trainer.train(
        epochs=config['transformer']['epochs'],
        patience=config['transformer']['patience'],
        save_dir=config['model_dir'],
        save_prefix='transformer'
    )
    
    # Vẽ biểu đồ lịch sử huấn luyện
    plot_training_history(
        history,
        save_path=os.path.join(config['model_dir'], 'transformer_history.png')
    )
    
    # Tính BLEU score
    bleu = calculate_bleu(
        model=model,
        test_loader=test_loader,
        src_sp=src_sp,
        tgt_sp=tgt_sp,
        device=config['device'],
        model_type='transformer',
        max_len=config['max_len']
    )
    
    logger.info(f"BLEU score của mô hình Transformer: {bleu:.2f}")
    
    # Lưu BLEU score
    with open(os.path.join(config['model_dir'], 'transformer_bleu.txt'), 'w') as f:
        f.write(f"BLEU score: {bleu:.2f}")
    
    return model, history


def train_lstm(config):
    """
    Huấn luyện mô hình LSTM
    
    Args:
        config (dict): Cấu hình
        
    Returns:
        tuple: (model, history)
    """
    # Tạo DataLoader
    train_loader, val_loader, test_loader, src_sp, tgt_sp = create_dataloaders(config)
    
    # Xác định kích thước từ điển
    src_vocab_size = src_sp.get_piece_size()
    tgt_vocab_size = tgt_sp.get_piece_size()
    
    # Tạo mô hình
    from lstm_model import create_lstm_model
    
    model = create_lstm_model(config['lstm'], src_vocab_size, tgt_vocab_size)
    model = model.to(config['device'])
    
    # Tạo optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['lstm']['learning_rate']
    )
    
    # Tạo scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        verbose=True
    )
    
    # Tạo trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=None,  # Sử dụng criterion trong mô hình
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
        patience=config['lstm']['patience'],
        save_dir=config['model_dir'],
        save_prefix='lstm'
    )
    
    # Vẽ biểu đồ lịch sử huấn luyện
    plot_training_history(
        history,
        save_path=os.path.join(config['model_dir'], 'lstm_history.png')
    )
    
    # Tính BLEU score
    bleu = calculate_bleu(
        model=model,
        test_loader=test_loader,
        src_sp=src_sp,
        tgt_sp=tgt_sp,
        device=config['device'],
        model_type='lstm',
        max_len=config['max_len']
    )
    
    logger.info(f"BLEU score của mô hình LSTM: {bleu:.2f}")
    
    # Lưu BLEU score
    with open(os.path.join(config['model_dir'], 'lstm_bleu.txt'), 'w') as f:
        f.write(f"BLEU score: {bleu:.2f}")
    
    return model, history


def compare_models(transformer_history, lstm_history, config):
    """
    So sánh hiệu suất của hai mô hình
    
    Args:
        transformer_history (dict): Lịch sử huấn luyện của mô hình Transformer
        lstm_history (dict): Lịch sử huấn luyện của mô hình LSTM
        config (dict): Cấu hình
    """
    # Tạo figure với 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot loss
    ax1.plot(transformer_history['train_loss'], label='Transformer Train Loss', marker='o')
    ax1.plot(transformer_history['val_loss'], label='Transformer Val Loss', marker='s')
    ax1.plot(lstm_history['train_loss'], label='LSTM Train Loss', marker='^')
    ax1.plot(lstm_history['val_loss'], label='LSTM Val Loss', marker='d')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss Comparison')
    ax1.legend()
    ax1.grid(True)
    
    # Plot perplexity
    ax2.plot(transformer_history['train_ppl'], label='Transformer Train PPL', marker='o')
    ax2.plot(transformer_history['val_ppl'], label='Transformer Val PPL', marker='s')
    ax2.plot(lstm_history['train_ppl'], label='LSTM Train PPL', marker='^')
    ax2.plot(lstm_history['val_ppl'], label='LSTM Val PPL', marker='d')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perplexity')
    ax2.set_title('Training and Validation Perplexity Comparison')
    ax2.legend()
    ax2.grid(True)
    
    # Điều chỉnh layout
    plt.tight_layout()
    
    # Lưu biểu đồ
    plt.savefig(os.path.join(config['model_dir'], 'model_comparison.png'))
    logger.info(f"Đã lưu biểu đồ so sánh mô hình tại {os.path.join(config['model_dir'], 'model_comparison.png')}")
    
    # Đọc BLEU score
    with open(os.path.join(config['model_dir'], 'transformer_bleu.txt'), 'r') as f:
        transformer_bleu = float(f.read().split(':')[1].strip())
    
    with open(os.path.join(config['model_dir'], 'lstm_bleu.txt'), 'r') as f:
        lstm_bleu = float(f.read().split(':')[1].strip())
    
    # Tạo bảng so sánh
    comparison = {
        'Model': ['Transformer', 'LSTM'],
        'Best Val Loss': [min(transformer_history['val_loss']), min(lstm_history['val_loss'])],
        'Best Val PPL': [min(transformer_history['val_ppl']), min(lstm_history['val_ppl'])],
        'BLEU Score': [transformer_bleu, lstm_bleu],
        'Training Time (s/epoch)': [np.mean(transformer_history['train_time']), np.mean(lstm_history['train_time'])]
    }
    
    comparison_df = pd.DataFrame(comparison)
    
    # Lưu bảng so sánh
    comparison_df.to_csv(os.path.join(config['model_dir'], 'model_comparison.csv'), index=False)
    logger.info(f"Đã lưu bảng so sánh mô hình tại {os.path.join(config['model_dir'], 'model_comparison.csv')}")
    
    # In bảng so sánh
    logger.info("\nSo sánh hiệu suất của hai mô hình:")
    logger.info(comparison_df.to_string(index=False))


if __name__ == "__main__":
    # Cấu hình mặc định
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
    
    # So sánh hiệu suất của hai mô hình
    compare_models(transformer_history, lstm_history, config)
