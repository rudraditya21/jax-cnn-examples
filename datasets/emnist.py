from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .registry import DatasetRegistry

EMNIST_MEAN = (0.1751,)
EMNIST_STD = (0.3332,)


def _build_transforms(augment: bool) -> tuple[transforms.Compose, transforms.Compose]:
    """
    Constructs the training ans evaluation transforms for EMNIST dataset.

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
                transforms.Normalize(EMNIST_MEAN, EMNIST_STD),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(EMNIST_MEAN, EMNIST_STD),
            ]
        )

    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(EMNIST_MEAN, EMNIST_STD),
        ]
    )

    return train_transform, eval_transform


@DatasetRegistry.register("emnist")
def load_emnist(
    data_dir: str = "./data",
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    augment: bool = True,
    split: str = "balanced",
) -> tuple[DataLoader, DataLoader, int]:
    """
    Loads the EMNIST dataset with standardized transforms and DataLoaders.

    The function provides a clean, unified entry point for integrating EMNIST into training pipelines. All dataset configuration parameters are exposed as function arguments to support experimentation and reproducibility.

    Args:
        data_dir (str, optional): Directory where the dataset is stored or will be downloaded. Defaults to "./data".
        batch_size (int, optional): Number of samples per batch. Defaults to 64.
        shuffle (bool, optional): Whether to shuffle the training dataset. Defaults to True.
        num_workers (int, optional): Number of worker subprocesses used by the DataLoader. Defaults to 4.
        augment (bool, optional): Whether to apply data augmentation to the training set. Defaults to True.
        split (str, optional): EMNIST split to load. Supported:
            - "balanced": 47 classes (digits + letters, balanced distribution)
            - "byclass": 62 classes (uppercase + lowercase + digits)
            - "bymerge": 47 classes (uppercase merged with lowercase)
            - "digits": 10 classes (digit-only subset)
            - "mnist": 10 classes (classic MNIST split)
            - "letters": 26 classes (uppercase letters only)

    Returns:
        Tuple[DataLoader, DataLoader, int]:
        A tuple containing:
            - The training DataLoader
            - The evaluation DataLoader
            - The number of output classes (10 for EMNIST)
    """
    data_path = Path(data_dir)
    train_transform, eval_transform = _build_transforms(augment)

    train_dataset = datasets.EMNIST(
        root=data_path,
        split=split,
        train=True,
        download=True,
        transform=train_transform,
    )

    eval_dataset = datasets.EMNIST(
        root=data_path,
        split=split,
        train=False,
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

    if split == "balanced":
        num_classes = 47
    elif split == "byclass":
        num_classes = 62
    elif split in ("digits", "mnist"):
        num_classes = 10
    elif split == "letters":
        num_classes = 26
    elif split == "bymerge":
        num_classes = 47
    else:
        raise ValueError(f"Unknown EMNIST split: {split}")

    return train_loader, eval_loader, num_classes
