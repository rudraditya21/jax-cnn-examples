from . import (
    cifar10,
    cifar100,
    country211,
    emnist,
    eurosat,
    fashion_mnist,
    fer2013,
    gtsrb,
    kmnist,
    mnist,
    pcam,
    qmnist,
    sun397,
)
from .registry import DatasetRegistry

__all__ = [
    "DatasetRegistry",
    "cifar10",
    "cifar100",
    "country211",
    "emnist",
    "eurosat",
    "fashion_mnist",
    "fer2013",
    "gtsrb",
    "kmnist",
    "mnist",
    "pcam",
    "qmnist",
    "sun397",
]
