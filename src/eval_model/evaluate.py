# src/evaluate.py
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

from src.logger import get_logger
from src.model import build_model
from src.model_metrics.metrics import compute_metrics
from src.exceptions import EvaluationError

logger = get_logger(__name__)

def evaluate(config: dict, model_path: str, data_loader, device: torch.device,
             idx_to_label: dict, output_dir: str = None):
    """
    Run full evaluation on a dataset.

    Args:
        config: project configuration dict.
        model_path: path to the saved model weights (.pth).
        data_loader: DataLoader for the evaluation set.
        device: torch device.
        idx_to_label: dict mapping class index to style name.
        output_dir: directory to save plots and report (defaults to config paths).

    Returns:
        metrics dict (as from compute_metrics).
    """
    if output_dir is None:
        output_dir = Path(config['paths']['plots_dir'])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    num_classes = config['model']['num_classes']
    model = build_model(config, num_classes).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f"Model loaded from {model_path}")

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.append(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.concatenate(all_probs, axis=0)

    # Compute metrics
    k = config['evaluation']['top_k']
    metrics = compute_metrics(all_labels, all_preds, all_probs, idx_to_label, k)
    top1 = metrics['top1_acc'] * 100
    topk = metrics[f'top{k}_acc'] * 100
    logger.info(f"Top-1 Accuracy: {top1:.2f}%")
    logger.info(f"Top-{k} Accuracy: {topk:.2f}%")

    # Classification report
    target_names = [idx_to_label[i] for i in range(num_classes)]
    report = classification_report(all_labels, all_preds, target_names=target_names, digits=3)
    logger.info("Classification Report:\n" + report)
    with open(output_dir / 'classification_report.txt', 'w') as f:
        f.write(report)

    # Confusion matrix plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(metrics['confusion_matrix'], annot=False, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved to {output_dir / 'confusion_matrix.png'}")

    # Per-class accuracy bar chart
    per_class = metrics['per_class_acc'] * 100
    sorted_idx = np.argsort(per_class)
    sorted_names = [target_names[i] for i in sorted_idx]
    sorted_vals = per_class[sorted_idx]
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(sorted_names)), sorted_vals, color='skyblue')
    plt.yticks(range(len(sorted_names)), sorted_names, fontsize=8)
    plt.xlabel('Accuracy (%)')
    plt.title('Per-Class Accuracy (sorted)')
    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_accuracy.png', dpi=150)
    plt.close()
    logger.info(f"Per-class accuracy plot saved to {output_dir / 'per_class_accuracy.png'}")

    return metrics