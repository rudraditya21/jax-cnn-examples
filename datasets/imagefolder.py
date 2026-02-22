from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .registry import DatasetRegistry

IMAGEFOLDER_MEAN = (0.485, 0.456, 0.406)
IMAGEFOLDER_STD = (0.229, 0.224, 0.225)


def _build_transforms(
    augment: bool,
    image_size: int,
) -> tuple[transforms.Compose, transforms.Compose]:
    """
    Constructs the training and evaluation transforms for an ImageFolder dataset.

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
                transforms.Normalize(IMAGEFOLDER_MEAN, IMAGEFOLDER_STD),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGEFOLDER_MEAN, IMAGEFOLDER_STD),
            ]
        )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGEFOLDER_MEAN, IMAGEFOLDER_STD),
        ]
    )

    return train_transform, eval_transform


@DatasetRegistry.register("imagefolder")
def load_imagefolder(
    data_dir: str = "./data",
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    augment: bool = True,
    image_size: int = 224,
    train_split: str = "train",
    eval_split: str = "val",
) -> tuple[DataLoader, DataLoader, int]:
    """
    Loads a folder-structured classification dataset with ImageFolder.

    The expected directory layout is:
    `data_dir/train/<class_dirs>` and `data_dir/val/<class_dirs>`.

    Args:
        data_dir (str, optional): Directory containing split folders. Defaults to "./data".
        batch_size (int, optional): Number of samples per batch. Defaults to 64.
        shuffle (bool, optional): Whether to shuffle the training dataset. Defaults to True.
        num_workers (int, optional): Number of worker subprocesses used by the DataLoader.
            Defaults to 4.
        augment (bool, optional): Whether to apply data augmentation to the training set.
            Defaults to True.
        image_size (int, optional): Spatial size to resize/crop images to. Defaults to 224.
        train_split (str, optional): Folder name to use for training. Defaults to "train".
        eval_split (str, optional): Folder name to use for evaluation. Defaults to "val".

    Returns:
        Tuple[DataLoader, DataLoader, int]:
        A tuple containing:
            - The training DataLoader
            - The evaluation DataLoader
            - The number of output classes inferred from the training set
    """
    data_path = Path(data_dir)
    train_path = data_path / train_split
    eval_path = data_path / eval_split

    if not train_path.exists() or not train_path.is_dir():
        raise FileNotFoundError(f"ImageFolder train split directory not found: {train_path}")
    if not eval_path.exists() or not eval_path.is_dir():
        raise FileNotFoundError(f"ImageFolder eval split directory not found: {eval_path}")

    train_transform, eval_transform = _build_transforms(augment, image_size)

    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    eval_dataset = datasets.ImageFolder(root=eval_path, transform=eval_transform)
    if train_dataset.class_to_idx != eval_dataset.class_to_idx:
        raise ValueError(
            "ImageFolder train/eval class mappings do not match. Ensure both splits use "
            "the same class directory names."
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

    num_classes = len(train_dataset.classes)

    return train_loader, eval_loader, num_classes
