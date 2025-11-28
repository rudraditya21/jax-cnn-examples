from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .registry import DatasetRegistry

QMNIST_MEAN = (0.1307,)
QMNIST_STD = (0.3081,)


def _build_transforms(augment: bool) -> tuple[transforms.Compose, transforms.Compose]:
    """
    Constructs the training ans evaluation transforms for QMNIST dataset.

    Args:
        augmet (bool): If True, data augmentation operations are applied to the training set. Evaluation transforms remain deterministic.

    Returns:
        Tuple[transforms.Compose, transforms.Compose]:
        A tuple containing the training transform and the evaluation transform.
    """
    if augment:
        train_transform = transforms.Compose(
            [
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(QMNIST_MEAN, QMNIST_STD),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(QMNIST_MEAN, QMNIST_STD),
            ]
        )

    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(QMNIST_MEAN, QMNIST_STD),
        ]
    )

    return train_transform, eval_transform


@DatasetRegistry.register("qmnist")
def load_qmnist(
    data_dir: str = "./data",
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    augment: bool = True,
    split: str = "test",
) -> tuple[DataLoader, DataLoader, int]:
    """
    Loads the QMNIST dataset with standardized transforms and DataLoaders.

    The function provides a clean, unified entry point for integrating QMNIST into training pipelines. All dataset configuration parameters are exposed as function arguments to support experimentation and reproducibility.

    Args:
        data_dir (str, optional): Directory where the dataset is stored or will be downloaded. Defaults to "./data".
        batch_size (int, optional): Number of samples per batch. Defaults to 64.
        shuffle (bool, optional): Whether to shuffle the training dataset. Defaults to True.
        num_workers (int, optional): Number of worker subprocesses used by the DataLoader. Defaults to 4.
        augment (bool, optional): Whether to apply data augmentation to the training set. Defaults to True.
        split (str): QMNIST split to load. Supported:
            - "train": 60k training images
            - "test": 10k QMNIST test replacement
            - "nist": full extended test set (60k)
            - "all": train + test (70k)

    Returns:
        Tuple[DataLoader, DataLoader, int]:
        A tuple containing:
            - The training DataLoader
            - The evaluation DataLoader
            - The number of output classes (10 for QMNIST)
    """
    data_path = Path(data_dir)
    train_transform, eval_transform = _build_transforms(augment)

    train_dataset = datasets.QMNIST(
        root=data_path,
        what=split if split in ("train", "all") else "train",
        download=True,
        transform=train_transform,
    )

    eval_dataset = datasets.QMNIST(
        root=data_path,
        what=split,
        download=True,
        transform=eval_transform,
    )

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
