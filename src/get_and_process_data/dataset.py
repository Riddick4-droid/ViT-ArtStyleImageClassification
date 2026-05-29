import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.exceptions import DatasetError

logger = get_logger(__name__)

def is_valid_image(path:str)->bool:
    """check to see whther an image file can be opened and is valid, without loading fully"""
    try:
        with Image.open(path) as img:
            img.verify() #this is a light verification (not a full decoding)
        #reopen the image to ensure full loading works-bcos verify does not always catch truncated images
        with Image.open(path) as img:
            img.load()
        return True
    except Exception:
        return False

def pil_loader(path:str)->Image.Image:
    """loads the image in RGB; this should only be called on validated(True) images"""
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except Exception as e:
        raise DatasetError(f"failed to load a previously validated image: {path} - {e}")
    
class ArtStyleDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path,label = self.samples[idx]
        image = pil_loader(path)
        if self.transform:
            image = self.transform(image)
        return image, label
    
def build_data_loader(config:dict):
    data_root = Path(config["data"]["local_dir"])
    if not data_root.exists():
        raise DatasetError(f"Data dir not found: {data_root}")
    
    #discover style classes
    style_dirs = sorted([d for d in data_root.iterdir() if d.is_dir() and not d.name.startswith(".")])
    if not style_dirs:
        raise DatasetError(f"no style folders found in {data_root}")
    style_names = [d.name for d in style_dirs]
    label_to_idx = {name:i for i, name in enumerate(style_names)}
    idx_to_label = {i:name for name,i in label_to_idx.items()}
    num_classes = len(style_names)
    logger.info(f"Found {num_classes} art styles: {style_names}")

    #build samplelist, filter out corrupted images
    all_samples = []
    skipped = 0
    for style_dir in style_dirs:
        label = label_to_idx[style_dir.name]
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff'):
            for img_path in style_dir.glob(ext):
                path_str = str(img_path)
                if is_valid_image(path=path_str):
                    all_samples.append((path_str,label))
                else:
                    skipped += 1
    if skipped:
        logger.warning(f"skipped {skipped} corrupted/unreasable images")
    if not all_samples:
        raise DatasetError("no valid images found in dataset")
    logger.info(f"total valid images: {len(all_samples)}")

    #stratified split
    labels = [label for _,label in all_samples]
    train_samples, val_samples = train_test_split(
        all_samples,
        test_size = config["data"]["val_split"],
        random_state=config["data"]["random_seed"],
        stratify=labels
    )
    logger.info(f"train samples: {len(train_samples)}, val samples: {len(val_samples)}")

    #transform
    img_size = config["data"]["image_size"]
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #dataset and data loaders
    train_dataset =ArtStyleDataset(samples=train_samples, transform=train_transforms)
    val_dataset = ArtStyleDataset(samples=val_samples, transform=val_transforms)

    #loader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=False
    )

    return train_loader, val_loader, num_classes, idx_to_label

