# src/trainer.py
import os
import copy
import time
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.cuda.amp import GradScaler, autocast

from src.logger import get_logger
from src.utils import get_model_info
from src.model_metrics.metrics import accuracy
from src.exceptions import TrainingError

logger = get_logger(__name__)

def build_optimizer(model, config):
    opt_cfg = config['training']
    if opt_cfg['optimizer'].lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=opt_cfg['learning_rate'],
                           weight_decay=opt_cfg['weight_decay'])
    elif opt_cfg['optimizer'].lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=opt_cfg['learning_rate'],
                         momentum=0.9, weight_decay=opt_cfg['weight_decay'])
    else:
        raise TrainingError(f"Unsupported optimizer: {opt_cfg['optimizer']}")

def build_scheduler(optimizer, config):
    sched_cfg = config['training']
    if sched_cfg['scheduler'] == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=sched_cfg['t_max'], eta_min=sched_cfg['eta_min'])
    elif sched_cfg['scheduler'] == 'step':
        return StepLR(optimizer, step_size=5, gamma=0.5)
    elif sched_cfg['scheduler'] == 'none':
        return None
    else:
        raise TrainingError(f"Unsupported scheduler: {sched_cfg['scheduler']}")

def train_model(config, model, train_loader, val_loader, criterion, device):
    """
    Full training loop.
    Returns best model state_dict, history dict.
    """
    epochs = config['training']['epochs']
    mixed_precision = config['training'].get('mixed_precision', False)
    save_best_only = config['training'].get('save_best_only', True)
    artefacts_dir = Path(config['paths']['artefacts'])
    models_dir = Path(config['paths']['models_dir'])
    models_dir.mkdir(parents=True, exist_ok=True)

    # Log model info
    info = get_model_info(model)
    logger.info(f"Model: {info['model_class']} – Total params: {info['total_params']:,}, "
                f"Trainable: {info['trainable_params']:,}")

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = GradScaler(enabled=mixed_precision)

    # Tracking
    best_val_acc = 0.0
    best_epoch = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # TensorBoard (optional)
    use_tb = config.get('use_tensorboard', False)
    if use_tb:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=config['paths']['logs_dir'])
        logger.info("TensorBoard logging enabled.")
    else:
        writer = None

    for epoch in range(epochs):
        logger.info(f"{'='*50} Epoch {epoch+1}/{epochs} {'='*50}")

        # ---------- Training ----------
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        start_time = time.time()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)

            with autocast(enabled=mixed_precision):
                outputs = model(images)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Metrics
            train_loss += loss.item() * images.size(0)
            batch_acc = accuracy(outputs, labels)
            train_correct += int(batch_acc * images.size(0))
            train_total += images.size(0)

            train_pbar.set_postfix({
                "loss":f"{loss.item():.3f}",
                "acc":f"{batch_acc:.3f}"
            })

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # ---------- Validation ----------
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                acc = accuracy(outputs, labels)
                val_correct += int(acc * images.size(0))
                val_total += images.size(0)

                val_pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{batch_acc:.4f}"
                })
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        elapsed = time.time() - start_time

        # Logging
        logger.info(f"Train Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc*100:.2f}%")
        logger.info(f"Val   Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc*100:.2f}% | "
                    f"Time: {elapsed:.0f}s")
        if scheduler is not None:
            logger.info(f"Learning rate: {scheduler.get_last_lr()[0]:.2e}")

        # Overfitting check
        if epoch_val_loss > 1.5 * epoch_train_loss:
            logger.warning("Validation loss significantly higher than train loss – possible overfitting.")

        # History
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        # TensorBoard
        if writer:
            writer.add_scalar('Loss/train', epoch_train_loss, epoch)
            writer.add_scalar('Loss/val', epoch_val_loss, epoch)
            writer.add_scalar('Accuracy/train', epoch_train_acc, epoch)
            writer.add_scalar('Accuracy/val', epoch_val_acc, epoch)

        # Scheduler step
        if scheduler:
            scheduler.step()

        # Checkpointing
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_epoch = epoch + 1
            best_model_wts = copy.deepcopy(model.state_dict())
            if save_best_only:
                torch.save(best_model_wts, models_dir / 'best_model.pth')
                logger.info(f"Best model saved (epoch {best_epoch}, val_acc={best_val_acc*100:.2f}%)")

        # Save latest (if not only best)
        if not save_best_only:
            torch.save(model.state_dict(), models_dir / f'epoch_{epoch+1}.pth')

    # Save final best if not already saved
    if save_best_only:
        torch.save(best_model_wts, models_dir / 'best_model.pth')

    if writer:
        writer.close()

    logger.info(f"Training complete. Best validation accuracy: {best_val_acc*100:.2f}% (epoch {best_epoch})")
    return best_model_wts, history