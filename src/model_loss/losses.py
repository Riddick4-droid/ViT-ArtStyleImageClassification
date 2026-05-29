import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from src.exceptions import ConfigurationError

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    Reduces overconfidence by mixing target distribution with a uniform prior.
    """
    def __init__(self, smoothing:float=0.1, weight:torch.Tensor=None):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.weight = weight

    def forward(self, pred,target):
        log_probs = torch.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss

        if self.weight is not None:
            #apply a per sample weight
            weight_per_sample = self.weight[target]
            loss = loss * weight_per_sample
        return loss.mean()
    
class FocalLoss(nn.Module):
    """
    Focal loss for addressing class imbalance.
    alpha: class weight (tensor of shape (num_classes,))
    gamma: focusing parameter (higher reduces loss for well-classified examples)
    """
    def __init__(self, alpha:torch.Tensor=None, gamma: float=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self,pred,target):
        log_prob = F.log_softmax(pred,dim=-1)
        prob = torch.exp(log_prob)
        target = target.view(-1,1)
        log_prob = log_prob.gather(1,target).view(-1)
        prob = prob.gather(1,target).view(-1)
        loss = - (1-prob) ** self.gamma * log_prob
        if self.alpha is not None:
            alpha_per_sample = self.alpha[target.view(-1)]
            loss = alpha_per_sample * loss
        return loss.mean()
    
def compute_class_weights(train_labels:list,num_classes:int)->torch.Tensor:
    """Compute inverse-frequency class weights from label list."""
    counts = Counter(train_labels)
    total = len(train_labels)
    weights = [total / (num_classes * counts.get(c,1)) for c in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32)

def get_loss(config: dict, train_labels: list = None)->nn.Module:
    """
    Build the loss function according to config.

    Config keys used:
        loss.type: "cross_entropy", "label_smoothing", "focal"
        loss.label_smoothing: smoothing factor
        loss.focal_alpha: alpha value for focal loss
        loss.focal_gamma: gamma value for focal loss
        loss.class_weight: "inverse_frequency" / "none"

    Args:
        config: full config dict
        train_labels: list of integer labels (needed if class_weight is 'inverse_frequency')
    Returns:
        PyTorch loss module
    """
    loss_cfg = config['loss']
    loss_type = loss_cfg.get('type', 'cross_entropy')
    class_weight_strategy = loss_cfg.get('class_weight', 'none')

    # Build class weights
    class_weights = None
    if class_weight_strategy == 'inverse_frequency':
        if train_labels is None:
            raise ConfigurationError("train_labels must be provided for inverse_frequency class weighting")
        num_classes = config['model']['num_classes']
        class_weights = compute_class_weights(train_labels, num_classes)

    if loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(weight=class_weights)

    elif loss_type == 'label_smoothing':
        smoothing = loss_cfg.get('label_smoothing', 0.1)
        return LabelSmoothingCrossEntropy(smoothing=smoothing, weight=class_weights)

    elif loss_type == 'focal':
        gamma = loss_cfg.get('focal_gamma', 2.0)
        alpha = class_weights  # can be overridden by explicit focal_alpha if desired
        return FocalLoss(alpha=alpha, gamma=gamma)

    else:
        raise ConfigurationError(f"Unknown loss type: {loss_type}")