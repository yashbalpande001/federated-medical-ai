import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.env_config import get_paths, load_compute_budget


class RSNADataset(Dataset):
    """
    PyTorch Dataset for RSNA Pneumonia Detection Chest X-Rays.
    Aggregates patient-level labels and supports DICOM / PNG image formats.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: Path,
        transform: Optional[transforms.Compose] = None,
        is_synthetic: bool = False,
        image_size: Tuple[int, int] = (224, 224),
    ):
        """
        Args:
            df: DataFrame containing aggregated patient metadata ('patientId', 'Target', 'bboxes').
            images_dir: Directory containing DICOM (.dcm) or PNG/JPG X-ray images.
            transform: torchvision transforms to apply to the images.
            is_synthetic: If True, generate synthetic X-ray tensors when files do not exist.
            image_size: Target image size tuple (height, width).
        """
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.is_synthetic = is_synthetic
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        patient_id = row["patientId"]
        target = int(row["Target"])

        if self.is_synthetic:
            image_tensor = self._generate_synthetic_image()
        else:
            image_tensor = self._load_image(patient_id)

        if self.transform is not None and not self.is_synthetic:
            # If transform is PIL/Tensor based
            image_tensor = self.transform(image_tensor)

        return image_tensor, target

    def _generate_synthetic_image(self) -> torch.Tensor:
        """Generates a synthetic 3-channel X-ray image tensor [3, H, W]."""
        # Create normalized random tensor simulating X-ray intensities
        arr = np.random.normal(loc=0.5, scale=0.15, size=(self.image_size[0], self.image_size[1])).astype(np.float32)
        arr = np.clip(arr, 0.0, 1.0)
        # Stack grayscale to 3 channels
        arr_3ch = np.stack([arr] * 3, axis=0)  # Shape: [3, H, W]
        tensor = torch.from_numpy(arr_3ch)

        # Standard ImageNet normalization if required
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (tensor - mean) / std

    def _load_image(self, patient_id: str) -> torch.Tensor:
        """Loads image from DICOM or PNG format and returns a normalized 3-channel tensor."""
        dcm_path = self.images_dir / f"{patient_id}.dcm"
        png_path = self.images_dir / f"{patient_id}.png"
        jpg_path = self.images_dir / f"{patient_id}.jpg"

        if dcm_path.exists():
            try:
                import pydicom

                dcm = pydicom.dcmread(str(dcm_path))
                img_arr = dcm.pixel_array.astype(np.float32)
                # Rescale DICOM values to [0, 1]
                img_min, img_max = img_arr.min(), img_arr.max()
                if img_max > img_min:
                    img_arr = (img_arr - img_min) / (img_max - img_min)
                else:
                    img_arr = np.zeros_like(img_arr)
            except Exception as e:
                return self._generate_synthetic_image()
        elif png_path.exists() or jpg_path.exists():
            from PIL import Image

            img_file = png_path if png_path.exists() else jpg_path
            img_pil = Image.open(img_file).convert("L")
            img_arr = np.array(img_pil, dtype=np.float32) / 255.0
        else:
            # Fallback to synthetic image if raw file missing
            return self._generate_synthetic_image()

        # Resize image using torch interpolate if dimensions differ
        tensor_2d = torch.from_numpy(img_arr).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        if (tensor_2d.shape[2], tensor_2d.shape[3]) != self.image_size:
            tensor_2d = torch.nn.functional.interpolate(
                tensor_2d, size=self.image_size, mode="bilinear", align_corners=False
            )

        tensor_2d = tensor_2d.squeeze(0)  # [1, H, W]
        # Repeat to 3 channels
        tensor_3ch = tensor_2d.repeat(3, 1, 1)  # [3, H, W]

        # Apply ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (tensor_3ch - mean) / std


def parse_rsna_annotations(csv_path: Path) -> pd.DataFrame:
    """
    Parses RSNA stage_2_train_labels.csv and aggregates bounding boxes by patientId.
    Returns DataFrame with unique patientId, aggregated Target (binary 0/1), and bboxes list.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"RSNA labels file not found at {csv_path}")

    df_raw = pd.read_csv(csv_path)

    # Group by patientId
    grouped = []
    for pid, group in df_raw.groupby("patientId"):
        target = group["Target"].iloc[0]
        bboxes = []
        if target == 1:
            for _, row in group.iterrows():
                if not pd.isna(row["x"]):
                    bboxes.append([row["x"], row["y"], row["width"], row["height"]])
        grouped.append({"patientId": pid, "Target": int(target), "bboxes": bboxes})

    return pd.DataFrame(grouped)


def generate_synthetic_metadata(num_patients: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generates synthetic patient DataFrame for dry-runs and unit tests."""
    np.random.seed(seed)
    pids = [f"synthetic_patient_{i:04d}" for i in range(num_patients)]
    # Simulate ~30% positive pneumonia rate (matching RSNA distribution)
    targets = np.random.choice([0, 1], size=num_patients, p=[0.7, 0.3])
    records = []
    for pid, tgt in zip(pids, targets):
        bboxes = [[100, 100, 50, 50]] if tgt == 1 else []
        records.append({"patientId": pid, "Target": int(tgt), "bboxes": bboxes})
    return pd.DataFrame(records)


def split_rsna_dataset(
    df: pd.DataFrame,
    subset_size: Optional[int] = 6000,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Applies compute budget subset filtering and patient-disjoint train/val/test splitting.
    """
    np.random.seed(seed)
    unique_df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Apply compute budget subset filtering if active
    if subset_size is not None and len(unique_df) > subset_size:
        unique_df = unique_df.iloc[:subset_size].reset_index(drop=True)

    n_total = len(unique_df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_df = unique_df.iloc[:n_train].reset_index(drop=True)
    val_df = unique_df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test_df = unique_df.iloc[n_train + n_val :].reset_index(drop=True)

    return train_df, val_df, test_df


class CachedRSNADataset(Dataset):
    """
    PyTorch Dataset loading pre-processed uint8 tensor batches from disk.
    """
    def __init__(self, cache_files: List[Path]):
        self.images = []
        self.labels = []
        self.patient_ids = []

        for cf in sorted(cache_files):
            data = torch.load(str(cf), weights_only=False)
            self.images.append(data["images"])
            self.labels.append(data["labels"])
            self.patient_ids.extend(data["patient_ids"])

        if len(self.images) > 0:
            self.images = torch.cat(self.images, dim=0)
            self.labels = torch.cat(self.labels, dim=0)
        else:
            self.images = torch.empty((0, 3, 224, 224), dtype=torch.uint8)
            self.labels = torch.empty((0,), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_tensor = self.images[idx].float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        normalized = (img_tensor - mean) / std
        return normalized, int(self.labels[idx])


def get_rsna_dataloaders(
    config_path: Optional[Path] = None,
    budget_path: Optional[Path] = None,
    override_batch_size: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Union[int, str]]]:
    """
    Helper function to load RSNA dataset and return PyTorch DataLoaders.
    Checks for preprocessed batch cache first; falls back to raw loading if cache missing.

    Returns:
        (train_loader, val_loader, test_loader, metadata_summary)
    """
    paths = get_paths()
    config_dir = paths.config_root

    # Load configurations
    if config_path is None:
        config_path = config_dir / "rsna.yaml"
    with open(config_path, "r") as f:
        rsna_config = yaml.safe_load(f)

    try:
        compute_budget = load_compute_budget()
        active_split = compute_budget["dataset_split"]["active_split"]
        subset_size = compute_budget["dataset_split"]["subset_size"] if active_split == "subset" else None
        seed = compute_budget["dataset_split"].get("seed", 42)
    except Exception:
        active_split = "subset"
        subset_size = 6000
        seed = 42

    cache_dir = paths.output_root / "cache"
    train_cache_files = list(cache_dir.glob("train_batch_*.pt")) if cache_dir.exists() else []

    if len(train_cache_files) > 0:
        val_cache_files = list(cache_dir.glob("val_batch_*.pt"))
        test_cache_files = list(cache_dir.glob("test_batch_*.pt"))

        train_dataset = CachedRSNADataset(train_cache_files)
        val_dataset = CachedRSNADataset(val_cache_files)
        test_dataset = CachedRSNADataset(test_cache_files)

        batch_size = override_batch_size or rsna_config["dataloader"]["batch_size"]
        num_workers = 0 if (paths.environment == "local" and os.name == "nt") else rsna_config["dataloader"].get("num_workers", 0)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=rsna_config["dataloader"]["shuffle"], num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        summary = {
            "environment": paths.environment,
            "active_split": active_split,
            "mode": "Disk Cache (.pt)",
            "total_patients": len(train_dataset) + len(val_dataset) + len(test_dataset),
            "train_patients": len(train_dataset),
            "val_patients": len(val_dataset),
            "test_patients": len(test_dataset),
            "batch_size": batch_size,
        }
        return train_loader, val_loader, test_loader, summary

    # Fallback to loading from CSV/DICOMs if disk cache does not exist
    labels_csv = paths.rsna_labels_path
    images_dir = paths.rsna_images_dir
    splits_dir = paths.output_root / "splits"
    splits_csv = paths.output_root / "rsna_patient_splits.csv"

    is_synthetic = False
    if (splits_dir / "train.csv").exists() and (splits_dir / "val.csv").exists() and (splits_dir / "test.csv").exists():
        train_df = pd.read_csv(splits_dir / "train.csv")
        val_df = pd.read_csv(splits_dir / "val.csv")
        test_df = pd.read_csv(splits_dir / "test.csv")
        df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        if not labels_csv.exists():
            is_synthetic = True
    elif splits_csv.exists():
        splits_df = pd.read_csv(splits_csv)
        train_df = splits_df[splits_df["split"] == "train"].reset_index(drop=True)
        val_df = splits_df[splits_df["split"] == "val"].reset_index(drop=True)
        test_df = splits_df[splits_df["split"] == "test"].reset_index(drop=True)
        df = splits_df
        if not labels_csv.exists():
            is_synthetic = True
    elif labels_csv.exists():
        df = parse_rsna_annotations(labels_csv)
        train_df, val_df, test_df = split_rsna_dataset(
            df,
            subset_size=subset_size,
            train_ratio=rsna_config["split"].get("train_ratio", 0.70),
            val_ratio=rsna_config["split"].get("val_ratio", 0.15),
            test_ratio=rsna_config["split"].get("test_ratio", 0.15),
            seed=seed,
        )
    else:
        is_synthetic = True
        df = generate_synthetic_metadata(num_patients=subset_size or 500, seed=seed)
        train_df, val_df, test_df = split_rsna_dataset(
            df,
            subset_size=subset_size,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=seed,
        )



    batch_size = override_batch_size or rsna_config["dataloader"]["batch_size"]
    num_workers = rsna_config["dataloader"].get("num_workers", 0)

    # Disable num_workers on local Windows if needed to prevent multi-processing spawn overhead
    if paths.environment == "local" and os.name == "nt":
        num_workers = 0

    image_size = (rsna_config["dataset"]["image_size"], rsna_config["dataset"]["image_size"])

    train_dataset = RSNADataset(train_df, images_dir=images_dir, is_synthetic=is_synthetic, image_size=image_size)
    val_dataset = RSNADataset(val_df, images_dir=images_dir, is_synthetic=is_synthetic, image_size=image_size)
    test_dataset = RSNADataset(test_df, images_dir=images_dir, is_synthetic=is_synthetic, image_size=image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=rsna_config["dataloader"]["shuffle"],
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    summary = {
        "environment": paths.environment,
        "active_split": active_split,
        "is_synthetic": is_synthetic,
        "total_patients": len(df),
        "train_patients": len(train_df),
        "val_patients": len(val_df),
        "test_patients": len(test_df),
        "batch_size": batch_size,
    }

    return train_loader, val_loader, test_loader, summary


if __name__ == "__main__":
    print("--- RSNA Medical Dataset Dry Run ---")
    tr_loader, v_loader, te_loader, info = get_rsna_dataloaders()
    print("Metadata Summary:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    # Inspect first batch
    images, labels = next(iter(tr_loader))
    print(f"\nBatch Inspection:")
    print(f"  Images batch shape : {images.shape}")
    print(f"  Labels batch shape : {labels.shape}")
    print(f"  Label values       : {labels[:8].tolist()}")
    print(f"  Pixel mean/std     : {images.mean().item():.4f} / {images.std().item():.4f}")
