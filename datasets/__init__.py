from . import cifar10, cifar100, emnist, fashion_mnist, kmnist, mnist
from .registry import DatasetRegistry

__all__ = [
    "DatasetRegistry",
    "cifar10",
    "cifar100",
    "emnist",
    "fashion_mnist",
    "kmnist",
    "mnist",
]
