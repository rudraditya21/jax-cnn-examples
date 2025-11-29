from __future__ import annotations

from jax.example_libraries import stax

from .registry import ModuleRegistry


@ModuleRegistry.register("vgg11")
def vgg11(num_classes: int = 1000, dropout_rate: float | None = 0.5):
    """
    VGG-11 as in Simonyan & Zisserman (2014).
    """
    layers = [
        stax.Conv(64, (3, 3), padding="SAME"),
        stax.Relu,
        stax.MaxPool((2, 2), strides=(2, 2), padding="VALID"),
        stax.Conv(128, (3, 3), padding="SAME"),
        stax.Relu,
        stax.MaxPool((2, 2), strides=(2, 2), padding="VALID"),
        stax.Conv(256, (3, 3), padding="SAME"),
        stax.Relu,
        stax.Conv(256, (3, 3), padding="SAME"),
        stax.Relu,
        stax.MaxPool((2, 2), strides=(2, 2), padding="VALID"),
        stax.Conv(512, (3, 3), padding="SAME"),
        stax.Relu,
        stax.Conv(512, (3, 3), padding="SAME"),
        stax.Relu,
        stax.MaxPool((2, 2), strides=(2, 2), padding="VALID"),
        stax.Conv(512, (3, 3), padding="SAME"),
        stax.Relu,
        stax.Conv(512, (3, 3), padding="SAME"),
        stax.Relu,
        stax.MaxPool((2, 2), strides=(2, 2), padding="VALID"),
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


@ModuleRegistry.register("vgg11-bn")
def vgg11_bn(num_classes: int = 1000, dropout_rate: float | None = 0.5):
    """
    VGG-11 with BatchNorm after each conv.
    """
    layers = [
        stax.Conv(64, (3, 3), padding="SAME"),
        stax.BatchNorm(),
        stax.Relu,
        stax.MaxPool((2, 2), strides=(2, 2), padding="VALID"),
        stax.Conv(128, (3, 3), padding="SAME"),
        stax.BatchNorm(),
        stax.Relu,
        stax.MaxPool((2, 2), strides=(2, 2), padding="VALID"),
        stax.Conv(256, (3, 3), padding="SAME"),
        stax.BatchNorm(),
        stax.Relu,
        stax.Conv(256, (3, 3), padding="SAME"),
        stax.BatchNorm(),
        stax.Relu,
        stax.MaxPool((2, 2), strides=(2, 2), padding="VALID"),
        stax.Conv(512, (3, 3), padding="SAME"),
        stax.BatchNorm(),
        stax.Relu,
        stax.Conv(512, (3, 3), padding="SAME"),
        stax.BatchNorm(),
        stax.Relu,
        stax.MaxPool((2, 2), strides=(2, 2), padding="VALID"),
        stax.Conv(512, (3, 3), padding="SAME"),
        stax.BatchNorm(),
        stax.Relu,
        stax.Conv(512, (3, 3), padding="SAME"),
        stax.BatchNorm(),
        stax.Relu,
        stax.MaxPool((2, 2), strides=(2, 2), padding="VALID"),
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
