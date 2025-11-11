"""
Training loop with early stopping, learning rate scheduling, and metrics logging.
"""

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time


class Trainer:
    """
    Training engine with early stopping and comprehensive logging.
    """
    
    def __init__(self, model, device='cpu', use_focal=True, 
                 alpha=0.25, gamma=2.0):
        self.model = model.to(device)
        self.device = device
        
        # Loss function
        if use_focal:
            from model import FocalLoss
            self.loss_fn = FocalLoss(alpha=alpha, gamma=gamma)
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'gpu_mem': []
        }
    
    def setup_optimizer(self, lr=1e-4, optimizer='adam', weight_decay=0.001):
        """Setup optimizer with learning rate and weight decay."""
        params = self.model.parameters()
        
        if optimizer == 'adam':
            self.optimizer = Adam(params, lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = SGD(params, lr=lr, weight_decay=weight_decay, 
                               momentum=0.9)
        
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)
    
    def train_epoch(self, loader):
        """Train for one epoch and return loss/accuracy."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc='Train')
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(batch_x)
            loss = self.loss_fn(logits, batch_y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
            
            pbar.set_postfix({'loss': f'{total_loss / (total / 32):.4f}'})
        
        epoch_loss = total_loss / len(loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, loader):
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in tqdm(loader, desc='Val'):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                logits = self.model(batch_x)
                loss = self.loss_fn(logits, batch_y)
                
                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
        
        val_loss = total_loss / len(loader)
        val_acc = correct / total
        
        return val_loss, val_acc
    
    def train(self, train_loader, val_loader, num_epochs=50, 
              early_stop_patience=5, model_dir='./models'):
        """
        Full training loop with early stopping.
        Saves best model and logs metrics.
        """
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Log
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(round(train_loss, 4))
            self.history['val_loss'].append(round(val_loss, 4))
            self.history['train_acc'].append(round(train_acc, 4))
            self.history['val_acc'].append(round(val_acc, 4))
            
            if torch.cuda.is_available():
                mem_gb = round(torch.cuda.memory_allocated() / 1e9, 4)
                self.history['gpu_mem'].append(mem_gb)
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 
                          f"{model_dir}/best_model.pt")
            else:
                patience_counter += 1
            
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            # Learning rate scheduling
            self.scheduler.step()
        
        return self.history
    
    def get_history_df(self):
        """Convert history to DataFrame."""
        return pd.DataFrame(self.history)
