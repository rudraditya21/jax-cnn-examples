from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .registry import DatasetRegistry

EUROSAT_MEAN = (0.485, 0.456, 0.406)
EUROSAT_STD = (0.229, 0.224, 0.225)


def _build_transforms(
    augment: bool,
    image_size: int,
) -> tuple[transforms.Compose, transforms.Compose]:
    """
    Constructs the training and evaluation transforms for EuroSAT dataset.

    Args:
        augment (bool): If True, data augmentation operations are applied to the training set.
        image_size (int): Spatial size to resize/crop images to.

    Returns:
        Tuple[transforms.Compose, transforms.Compose]:
        A tuple containing the training transform and the evaluation transform.
    """
    if augment:
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomCrop(image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(EUROSAT_MEAN, EUROSAT_STD),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(EUROSAT_MEAN, EUROSAT_STD),
            ]
        )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(EUROSAT_MEAN, EUROSAT_STD),
        ]
    )

    return train_transform, eval_transform


@DatasetRegistry.register("eurosat")
def load_eurosat(
    data_dir: str = "./data",
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    augment: bool = True,
    image_size: int = 64,
    eval_split: float = 0.2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, int]:
    """
    Loads the EuroSAT dataset with standardized transforms and DataLoaders.

    EuroSAT does not ship with official train/test splits, so a deterministic random split
    is used to form the training and evaluation subsets.

    Args:
        data_dir (str, optional): Directory where the dataset is stored or will be downloaded.
            Defaults to "./data".
        batch_size (int, optional): Number of samples per batch. Defaults to 64.
        shuffle (bool, optional): Whether to shuffle the training dataset. Defaults to True.
        num_workers (int, optional): Number of worker subprocesses used by the DataLoader.
            Defaults to 4.
        augment (bool, optional): Whether to apply data augmentation to the training set.
            Defaults to True.
        image_size (int, optional): Spatial size to resize/crop images to. Defaults to 64.
        eval_split (float, optional): Fraction of samples reserved for evaluation.
            Defaults to 0.2.
        seed (int, optional): Random seed for deterministic splitting. Defaults to 42.

    Returns:
        Tuple[DataLoader, DataLoader, int]:
        A tuple containing:
            - The training DataLoader
            - The evaluation DataLoader
            - The number of output classes (10 for EuroSAT)
    """
    if not 0.0 < eval_split < 1.0:
        raise ValueError("eval_split must be between 0 and 1.")

    data_path = Path(data_dir)
    train_transform, eval_transform = _build_transforms(augment, image_size)

    base_dataset = datasets.EuroSAT(root=data_path, download=True)
    num_samples = len(base_dataset)
    eval_size = int(num_samples * eval_split)
    train_size = num_samples - eval_size
    if eval_size == 0 or train_size == 0:
        raise ValueError("eval_split results in an empty train or eval set.")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_samples, generator=generator).tolist()
    train_indices = indices[:train_size]
    eval_indices = indices[train_size:]

    train_dataset_full = datasets.EuroSAT(
        root=data_path,
        download=True,
        transform=train_transform,
    )
    eval_dataset_full = datasets.EuroSAT(
        root=data_path,
        download=True,
        transform=eval_transform,
    )

    train_dataset = Subset(train_dataset_full, train_indices)
    eval_dataset = Subset(eval_dataset_full, eval_indices)

    pin_memory = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    num_classes = 10

    return train_loader, eval_loader, num_classes
