from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .registry import DatasetRegistry

CELEBA_MEAN = (0.5063, 0.4258, 0.3832)
CELEBA_STD = (0.2669, 0.2452, 0.2414)


def _build_transforms(
    augment: bool,
    image_size: int,
) -> tuple[transforms.Compose, transforms.Compose]:
    """
    Constructs the training and evaluation transforms for CelebA dataset.

    Args:
        augment (bool): If True, data augmentation operations are applied to the training set.
        image_size (int): Spatial size to resize/crop images to.

    Returns:
        Tuple[transforms.Compose, transforms.Compose]:
        A tuple containing the training transform and the evaluation transform.
    """
    resize_size = image_size + 32
    if augment:
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CELEBA_MEAN, CELEBA_STD),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(CELEBA_MEAN, CELEBA_STD),
            ]
        )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(CELEBA_MEAN, CELEBA_STD),
        ]
    )

    return train_transform, eval_transform


def _to_int_target(target: object) -> int:
    if isinstance(target, tuple):
        if len(target) != 1:
            raise ValueError("CelebA identity target must contain exactly one value.")
        target = target[0]
    if isinstance(target, torch.Tensor):
        return int(target.item())
    return int(target)


def _infer_identity_offset(train_dataset: datasets.CelebA, eval_dataset: datasets.CelebA) -> int:
    train_min = int(train_dataset.identity.min().item())
    eval_min = int(eval_dataset.identity.min().item())
    min_identity = min(train_min, eval_min)
    if min_identity not in (0, 1):
        raise ValueError(f"Unexpected CelebA identity label minimum: {min_identity}")
    return min_identity


def _make_identity_target_transform(offset: int):
    def _transform(target: object) -> int:
        return _to_int_target(target) - offset

    return _transform


def _infer_num_classes(
    train_dataset: datasets.CelebA,
    eval_dataset: datasets.CelebA,
    offset: int,
) -> int:
    train_max = int(train_dataset.identity.max().item())
    eval_max = int(eval_dataset.identity.max().item())
    max_identity = max(train_max, eval_max)
    return max_identity - offset + 1


@DatasetRegistry.register("celeba")
def load_celeba(
    data_dir: str = "./data",
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    augment: bool = True,
    image_size: int = 224,
    train_split: str = "train",
    eval_split: str = "test",
) -> tuple[DataLoader, DataLoader, int]:
    """
    Loads the CelebA dataset with standardized transforms and DataLoaders.

    This loader uses identity labels (single-class classification) to stay compatible with
    the repository's single-label training loop.

    Args:
        data_dir (str, optional): Directory where the dataset is stored or will be downloaded.
            Defaults to "./data".
        batch_size (int, optional): Number of samples per batch. Defaults to 64.
        shuffle (bool, optional): Whether to shuffle the training dataset. Defaults to True.
        num_workers (int, optional): Number of worker subprocesses used by the DataLoader.
            Defaults to 4.
        augment (bool, optional): Whether to apply data augmentation to the training set.
            Defaults to True.
        image_size (int, optional): Spatial size to resize/crop images to. Defaults to 224.
        train_split (str, optional): Dataset split to use for training. Defaults to "train".
        eval_split (str, optional): Dataset split to use for evaluation. Defaults to "test".

    Returns:
        Tuple[DataLoader, DataLoader, int]:
        A tuple containing:
            - The training DataLoader
            - The evaluation DataLoader
            - The number of output classes inferred from identity labels
    """
    data_path = Path(data_dir)
    train_transform, eval_transform = _build_transforms(augment, image_size)

    train_dataset = datasets.CelebA(
        root=data_path,
        split=train_split,
        target_type="identity",
        download=True,
        transform=train_transform,
    )
    eval_dataset = datasets.CelebA(
        root=data_path,
        split=eval_split,
        target_type="identity",
        download=True,
        transform=eval_transform,
    )

    identity_offset = _infer_identity_offset(train_dataset, eval_dataset)
    target_transform = _make_identity_target_transform(identity_offset)
    train_dataset.target_transform = target_transform
    eval_dataset.target_transform = target_transform
    num_classes = _infer_num_classes(train_dataset, eval_dataset, identity_offset)

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

    return train_loader, eval_loader, num_classes
