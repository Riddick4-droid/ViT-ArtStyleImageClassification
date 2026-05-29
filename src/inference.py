# src/inference.py
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms

from src.logger import get_logger
from src.model.t_model import build_model
from src.exceptions import InferenceError

logger = get_logger(__name__)


def load_model_for_inference(config: dict, model_path: str, device: torch.device):
    """Load model from config and weights, set to eval mode."""
    num_classes = config['model']['num_classes']
    model = build_model(config, num_classes).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f"Model loaded from {model_path}")
    return model


def preprocess_image(image_path: str, image_size: int = 224) -> torch.Tensor:
    """Load and transform a single image for inference."""
    if not Path(image_path).is_file():
        raise InferenceError(f"Image file not found: {image_path}")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)  # add batch dimension
    return tensor


def predict(model, input_tensor: torch.Tensor, device: torch.device,
            idx_to_label: dict = None, top_k: int = 5):
    """
    Run inference on a preprocessed image tensor.

    Args:
        model: trained ViT model.
        input_tensor: (1, C, H, W) tensor.
        device: torch device.
        idx_to_label: mapping from class index to style name.
        top_k: number of top predictions to return.

    Returns:
        dict with keys:
            - 'predicted_class': class index
            - 'predicted_label': class name (if idx_to_label provided)
            - 'top_k': list of (label_name, probability) tuples
    """
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0)  # (num_classes,)

    # Top predicted class
    pred_idx = probs.argmax().item()
    confidence = probs[pred_idx].item()

    result = {
        'predicted_class': pred_idx,
        'confidence': confidence,
    }
    if idx_to_label:
        result['predicted_label'] = idx_to_label[pred_idx]

    # Top-k predictions
    topk_probs, topk_indices = torch.topk(probs, min(top_k, len(probs)))
    topk_list = []
    for i in range(len(topk_indices)):
        idx = topk_indices[i].item()
        prob = topk_probs[i].item()
        label = idx_to_label.get(idx, str(idx)) if idx_to_label else str(idx)
        topk_list.append((label, prob))
    result['top_k'] = topk_list

    return result