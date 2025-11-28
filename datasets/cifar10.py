from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .registry import DatasetRegistry

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def _build_transforms(augment: bool) -> tuple[transforms.Compose, transforms.Compose]:
    """
    Constructs the training and evaluation transforms for CIFAR-10 dataset.

    Args:
        augment (bool): If True, data augmentation operations are applied to the training set. Evaluation transforms remain deterministic.

    Returns:
        Tuple[transforms.Compose, transforms.Compose]:
        A tuple containing the training transform and the evaluation transform.
    """
    if augment:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    return train_transform, test_transform


@DatasetRegistry.register("cifar10")
def load_cifar10(
    data_dir: str = "./data",
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    augment: bool = True,
) -> tuple[DataLoader, DataLoader, int]:
    """
    Loads the CIFAR-10 dataset with standardized transforms and loaders.

    The function provides a clean, unified entry point for integrating CIFAR-10 into training pipelines. All dataset configuration parameters are exposed as function arguments to support experimentation and reproducibility.

    Args:
        data_dir (str, optional): Directory where the dataset is stored or will be downloaded. Defaults to "./data".
        batch_size (int, optional): Number of samples per batch. Defaults to 64.
        shuffle (bool, optional): Whether to shuffle the training dataset. Defaults to True.
        num_workers (int, optional): Number of worker subprocesses used by the DataLoader. Defaults to 4.
        augment (bool, optional): Whether to apply data augmentation to the training set. Defaults to True.

    Returns:
        Tuple[DataLoader, DataLoader, int]:
        A tuple containing:
            - The training DataLoader
            - The evaluation DataLoader
            - The number of output classes (10 for CIFAR-10)
    """
    data_path = Path(data_dir)
    train_transform, eval_transform = _build_transforms(augment)

    train_dataset = datasets.CIFAR10(
        root=data_path,
        train=False,
        download=True,
        transform=train_transform,
    )

    eval_dataset = datasets.CIFAR10(
        root=data_path,
        train=False,
        download=True,
        transform=eval_transform,
    )

    # pin_memory is enabled to improve host-to-GPU transfer performance.
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
