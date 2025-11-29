from __future__ import annotations

from jax.example_libraries import stax

from .registry import ModuleRegistry


@ModuleRegistry.register("alexnet")
def alexnet(num_classes: int = 1000, dropout_rate: float | None = 0.5):
    """
    AlexNet (2012) topology for ImageNet-sized inputs.
    Note: LRN is omitted because stax lacks a native operator; the rest matches the original ordering.
    """
    layers = [
        stax.Conv(64, (11, 11), strides=(4, 4), padding="SAME"),
        stax.Relu,
        stax.MaxPool((3, 3), strides=(2, 2), padding="VALID"),
        stax.Conv(192, (5, 5), padding="SAME"),
        stax.Relu,
        stax.MaxPool((3, 3), strides=(2, 2), padding="VALID"),
        stax.Conv(384, (3, 3), padding="SAME"),
        stax.Relu,
        stax.Conv(384, (3, 3), padding="SAME"),
        stax.Relu,
        stax.Conv(256, (3, 3), padding="SAME"),
        stax.Relu,
        stax.MaxPool((3, 3), strides=(2, 2), padding="VALID"),
        stax.Flatten,
        stax.Dense(4096),
        stax.Relu,
    ]

    if dropout_rate:
        layers.append(stax.Dropout(dropout_rate))

    layers.extend(
        [
            stax.Dense(4096),
            stax.Relu,
        ]
    )

    if dropout_rate:
        layers.append(stax.Dropout(dropout_rate))

    layers.append(stax.Dense(num_classes))

    return stax.serial(*layers)


@ModuleRegistry.register("alexnet-lite")
def alexnet_lite(num_classes: int = 10, dropout_rate: float | None = 0.5):
    """
    Channel-reduced AlexNet for 32x32 inputs (e.g., CIFAR).
    """
    layers = [
        stax.Conv(48, (11, 11), strides=(4, 4), padding="SAME"),
        stax.Relu,
        stax.MaxPool((3, 3), strides=(2, 2), padding="VALID"),
        stax.Conv(128, (5, 5), padding="SAME"),
        stax.Relu,
        stax.MaxPool((3, 3), strides=(2, 2), padding="VALID"),
        stax.Conv(256, (3, 3), padding="SAME"),
        stax.Relu,
        stax.Conv(256, (3, 3), padding="SAME"),
        stax.Relu,
        stax.Conv(192, (3, 3), padding="SAME"),
        stax.Relu,
        stax.MaxPool((3, 3), strides=(2, 2), padding="VALID"),
        stax.Flatten,
        stax.Dense(2048),
        stax.Relu,
    ]

    if dropout_rate:
        layers.append(stax.Dropout(dropout_rate))

    layers.extend(
        [
            stax.Dense(2048),
            stax.Relu,
        ]
    )

    if dropout_rate:
        layers.append(stax.Dropout(dropout_rate))

    layers.append(stax.Dense(num_classes))

    return stax.serial(*layers)
