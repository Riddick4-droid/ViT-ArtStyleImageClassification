# main.py
import argparse
import torch
from pathlib import Path
from src.logger import get_logger
from src.utils import load_config
from src.exceptions import ProjectException

logger = get_logger(__name__)

def cmd_ingest(config):
    from src.get_and_process_data.data_ingestion import download_and_cache
    download_and_cache(config['data']['dataset_name'], config['data']['local_dir'])

def cmd_train(config):
    import torch
    from src.get_and_process_data.dataset import build_data_loaders
    from src.model import build_model
    from src.model_loss.losses import get_loss
    from src.train_model.trainer import train_model
    from src.utils import get_model_info

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Data
    train_loader, val_loader, num_classes, idx_to_label = build_data_loaders(config)
    # Overwrite config's num_classes with actual count
    config['model']['num_classes'] = num_classes

    # Model
    model = build_model(config, num_classes).to(device)

    # Loss – need train labels for class weights
    train_labels = [lbl for _, lbl in train_loader.dataset.samples]
    criterion = get_loss(config, train_labels).to(device)

    # Train
    best_weights, history = train_model(config, model, train_loader, val_loader, criterion, device)
    logger.info("Training completed. Best model saved.")

def cmd_evaluate(config):
    import torch
    from src.get_and_process_data.dataset import build_data_loaders
    from src.eval_model.evaluate import evaluate
    from pathlib import Path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader, num_classes, idx_to_label = build_data_loaders(config)
    config['model']['num_classes'] = num_classes

    model_path = Path(config['paths']['models_dir']) / 'best_model.pth'
    if not model_path.exists():
        raise ProjectException(f"No best model found at {model_path}. Run training first.")
    evaluate(config, str(model_path), val_loader, device, idx_to_label)

def cmd_infer(config, image_path):
    import torch
    from src.inference import load_model_for_inference, preprocess_image, predict

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # We need label mapping; we can load it from dataset (or we can store it after training)
    # For simplicity, we rebuild data module temporarily to get mapping
    from src.get_and_process_data.dataset import build_data_loader
    _, _, num_classes, idx_to_label = build_data_loader(config)
    config['model']['num_classes'] = num_classes

    model_path = Path(config['paths']['models_dir']) / 'best_model.pth'
    if not model_path.exists():
        raise ProjectException(f"No best model found at {model_path}. Run training first.")

    model = load_model_for_inference(config, str(model_path), device)
    tensor = preprocess_image(image_path, config['data']['image_size'])
    result = predict(model, tensor, device, idx_to_label, config['evaluation']['top_k'])

    print(f"\nPredicted style: {result['predicted_label']} (confidence: {result['confidence']:.4f})")
    print("Top predictions:")
    for i, (label, prob) in enumerate(result['top_k'], 1):
        print(f"  {i}. {label}: {prob:.4f}")

def main():
    parser = argparse.ArgumentParser(description="ViT Art Style Classification")
    parser.add_argument('--config', default='configs/config.yaml', help='Path to config file')
    subparsers = parser.add_subparsers(dest='command', required=True, help='Sub-command to run')

    # ingest
    subparsers.add_parser('ingest', help='Download and cache dataset')

    # train
    subparsers.add_parser('train', help='Train the model')

    # evaluate
    subparsers.add_parser('evaluate', help='Evaluate the model on validation set')

    # infer
    infer_parser = subparsers.add_parser('infer', help='Predict art style for a single image')
    infer_parser.add_argument('image_path', help='Path to input image file')

    args = parser.parse_args()
    config = load_config(args.config)

    try:
        if args.command == 'ingest':
            cmd_ingest(config)
        elif args.command == 'train':
            cmd_train(config)
        elif args.command == 'evaluate':
            cmd_evaluate(config)
        elif args.command == 'infer':
            cmd_infer(config, args.image_path)
    except ProjectException as e:
        logger.error(f"Project error: {e}")
        exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        exit(1)

if __name__ == '__main__':
    main()



# python main.py --config configs/config.yaml ingest
# python main.py --config configs/config.yaml train
# python main.py --config configs/config.yaml evaluate
# python main.py --config configs/config.yaml infer path/to/painting.jpg