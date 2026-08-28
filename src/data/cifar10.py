import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src.utils.env_config import get_paths


def get_dataloaders(batch_size=64, val_split=0.1, data_dir=None, num_workers=0):
    """Data loaders for standard SimpleCNN (32x32 resolution)."""
    if data_dir is None:
        data_path = get_paths()["data_root"]
    else:
        data_path = Path(data_dir)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    full_train = datasets.CIFAR10(root=str(data_path), train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=str(data_path), train=False, download=True, transform=transform)

    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def get_dataloaders_resnet(batch_size=32, val_split=0.1, num_workers=2, data_dir=None):
    """
    Data loaders for ResNet models:
    - Resizes CIFAR-10 images to 224x224
    - Normalizes using ImageNet mean & std values
    """
    if data_dir is None:
        data_path = get_paths()["data_root"]
    else:
        data_path = Path(data_dir)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_train = datasets.CIFAR10(root=str(data_path), train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=str(data_path), train=False, download=True, transform=transform)

    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
