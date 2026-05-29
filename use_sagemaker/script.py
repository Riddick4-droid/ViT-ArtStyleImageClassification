# script.py
"""
SageMaker Training Script for ViT Art Style Classification.

This script is designed to be run inside a SageMaker training container.
Hyperparameters are passed as command‑line arguments (customary for PyTorch estimators).
Data is expected in the environment variable SM_CHANNEL_TRAINING.
The trained model is saved to SM_MODEL_DIR.
"""
import os
import sys
import argparse
from pathlib import Path

import torch

# Add project root to sys.path if needed
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.logger import get_logger
from src.utils import load_config
from src.get_and_process_data.dataset import build_data_loader
from src.model.t_model import build_model
from src.model_loss.losses import get_loss
from src.train_model.trainer import train_model

logger = get_logger(__name__, log_dir="/opt/ml/output/logs")  # SageMaker captures logs from here

def parse_args():
    parser = argparse.ArgumentParser(description="SageMaker ViT Training")
    # Hyperparameters (these match config keys; passed by SageMaker Estimator)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=3)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--config", default="configs/config.yaml", help="Base config file")
    # SageMaker specific
    parser.add_argument("--model_dir", default=os.environ.get("SM_MODEL_DIR", "./artefacts/models"))
    parser.add_argument("--train_data_dir", default=os.environ.get("SM_CHANNEL_TRAINING", "./data/wikiart"))
    return parser.parse_args()

def main():
    args = parse_args()

    # Load base config
    config = load_config(args.config)

    # Override config with SageMaker hyperparameters
    config['training']['epochs'] = args.epochs
    config['training']['learning_rate'] = args.learning_rate
    config['data']['batch_size'] = args.batch_size
    config['model']['embed_dim'] = args.embed_dim
    config['model']['depth'] = args.depth
    config['model']['num_heads'] = args.num_heads
    config['model']['patch_size'] = args.patch_size
    config['model']['dropout'] = args.dropout
    config['data']['local_dir'] = args.train_data_dir
    config['paths']['models_dir'] = args.model_dir

    # Ensure output directories exist
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(config['paths']['logs_dir'], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Data
    train_loader, val_loader, num_classes, idx_to_label = build_data_loader(config)
    config['model']['num_classes'] = num_classes

    # Model
    model = build_model(config, num_classes).to(device)

    # Loss
    train_labels = [lbl for _, lbl in train_loader.dataset.samples]
    criterion = get_loss(config, train_labels).to(device)

    # Train
    best_weights, history = train_model(config, model, train_loader, val_loader, criterion, device)
    logger.info("Training complete. Best model saved to {}".format(args.model_dir))

if __name__ == "__main__":
    main()