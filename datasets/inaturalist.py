from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .registry import DatasetRegistry

INATURALIST_MEAN = (0.485, 0.456, 0.406)
INATURALIST_STD = (0.229, 0.224, 0.225)


def _build_transforms(
    augment: bool,
    image_size: int,
) -> tuple[transforms.Compose, transforms.Compose]:
    """
    Constructs the training and evaluation transforms for INaturalist dataset.

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
                transforms.Normalize(INATURALIST_MEAN, INATURALIST_STD),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(INATURALIST_MEAN, INATURALIST_STD),
            ]
        )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(INATURALIST_MEAN, INATURALIST_STD),
        ]
    )

    return train_transform, eval_transform


def _to_int_target(target: object) -> int:
    if isinstance(target, tuple):
        if len(target) != 1:
            raise ValueError("INaturalist target tuple must contain exactly one value.")
        target = target[0]
    if isinstance(target, torch.Tensor):
        return int(target.item())
    return int(target)


def _infer_num_classes(dataset: object) -> int:
    if hasattr(dataset, "num_classes"):
        return int(dataset.num_classes)

    if hasattr(dataset, "all_categories"):
        categories = dataset.all_categories
        if isinstance(categories, dict):
            return len(categories)
        if hasattr(categories, "__len__"):
            return len(categories)
        if isinstance(categories, Iterable):
            return len(list(categories))

    if hasattr(dataset, "targets"):
        targets = dataset.targets
        if isinstance(targets, list) and targets:
            return len(set(int(t) for t in targets))

    raise ValueError("Unable to infer number of classes for INaturalist dataset.")


@DatasetRegistry.register("inaturalist")
def load_inaturalist(
    data_dir: str = "./data",
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    augment: bool = True,
    image_size: int = 224,
    train_version: str = "2021_train",
    eval_version: str = "2021_valid",
    target_type: str = "full",
) -> tuple[DataLoader, DataLoader, int]:
    """
    Loads the INaturalist dataset with standardized transforms and DataLoaders.

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
        train_version (str, optional): Dataset version to use for training.
            Defaults to "2021_train".
        eval_version (str, optional): Dataset version to use for evaluation.
            Defaults to "2021_valid".
        target_type (str, optional): Target hierarchy type. Defaults to "full".

    Returns:
        Tuple[DataLoader, DataLoader, int]:
        A tuple containing:
            - The training DataLoader
            - The evaluation DataLoader
            - The number of output classes inferred from the training set
    """
    data_path = Path(data_dir)
    train_transform, eval_transform = _build_transforms(augment, image_size)

    train_dataset = datasets.INaturalist(
        root=data_path,
        version=train_version,
        target_type=target_type,
        download=True,
        transform=train_transform,
        target_transform=_to_int_target,
    )
    eval_dataset = datasets.INaturalist(
        root=data_path,
        version=eval_version,
        target_type=target_type,
        download=True,
        transform=eval_transform,
        target_transform=_to_int_target,
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

    num_classes = _infer_num_classes(train_dataset)

    return train_loader, eval_loader, num_classes
