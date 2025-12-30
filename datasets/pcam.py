from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .registry import DatasetRegistry

PCAM_MEAN = (0.485, 0.456, 0.406)
PCAM_STD = (0.229, 0.224, 0.225)


def _build_transforms(
    augment: bool,
    image_size: int,
) -> tuple[transforms.Compose, transforms.Compose]:
    """
    Constructs the training and evaluation transforms for PCAM dataset.

    Args:
        augment (bool): If True, data augmentation operations are applied to the training set.
        image_size (int): Spatial size to resize images to.

    Returns:
        Tuple[transforms.Compose, transforms.Compose]:
        A tuple containing the training transform and the evaluation transform.
    """
    if augment:
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(PCAM_MEAN, PCAM_STD),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(PCAM_MEAN, PCAM_STD),
            ]
        )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(PCAM_MEAN, PCAM_STD),
        ]
    )

    return train_transform, eval_transform


@DatasetRegistry.register("pcam")
def load_pcam(
    data_dir: str = "./data",
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    augment: bool = True,
    image_size: int = 96,
    train_split: str = "train",
    eval_split: str = "test",
) -> tuple[DataLoader, DataLoader, int]:
    """
    Loads the PCAM dataset with standardized transforms and DataLoaders.

    Args:
        data_dir (str, optional): Directory where the dataset is stored or will be downloaded.
            Defaults to "./data".
        batch_size (int, optional): Number of samples per batch. Defaults to 64.
        shuffle (bool, optional): Whether to shuffle the training dataset. Defaults to True.
        num_workers (int, optional): Number of worker subprocesses used by the DataLoader.
            Defaults to 4.
        augment (bool, optional): Whether to apply data augmentation to the training set.
            Defaults to True.
        image_size (int, optional): Spatial size to resize images to. Defaults to 96.
        train_split (str, optional): Dataset split to use for training. Defaults to "train".
        eval_split (str, optional): Dataset split to use for evaluation. Defaults to "test".

    Returns:
        Tuple[DataLoader, DataLoader, int]:
        A tuple containing:
            - The training DataLoader
            - The evaluation DataLoader
            - The number of output classes (2 for PCAM)
    """
    data_path = Path(data_dir)
    train_transform, eval_transform = _build_transforms(augment, image_size)

    train_dataset = datasets.PCAM(
        root=data_path,
        split=train_split,
        download=True,
        transform=train_transform,
    )

    eval_dataset = datasets.PCAM(
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

    num_classes = 2

    return train_loader, eval_loader, num_classes
