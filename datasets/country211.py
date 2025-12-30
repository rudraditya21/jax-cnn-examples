from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .registry import DatasetRegistry

COUNTRY211_MEAN = (0.485, 0.456, 0.406)
COUNTRY211_STD = (0.229, 0.224, 0.225)


def _build_transforms(
    augment: bool,
    image_size: int,
) -> tuple[transforms.Compose, transforms.Compose]:
    """
    Constructs the training and evaluation transforms for Country211 dataset.

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
                transforms.Normalize(COUNTRY211_MEAN, COUNTRY211_STD),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(COUNTRY211_MEAN, COUNTRY211_STD),
            ]
        )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(COUNTRY211_MEAN, COUNTRY211_STD),
        ]
    )

    return train_transform, eval_transform


@DatasetRegistry.register("country211")
def load_country211(
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
    Loads the Country211 dataset with standardized transforms and DataLoaders.

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
            - The number of output classes (211 for Country211)
    """
    data_path = Path(data_dir)
    train_transform, eval_transform = _build_transforms(augment, image_size)

    train_dataset = datasets.Country211(
        root=data_path,
        split=train_split,
        download=True,
        transform=train_transform,
    )

    eval_dataset = datasets.Country211(
        root=data_path,
        split=eval_split,
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

    num_classes = 211

    return train_loader, eval_loader, num_classes
