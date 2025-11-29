from __future__ import annotations

from jax.example_libraries import stax

from .registry import ModuleRegistry


@ModuleRegistry.register("vgg19")
def vgg19(num_classes: int = 1000, dropout_rate: float | None = 0.5):
    """
    VGG-19 configuration (E).
    """
    layers = [
        stax.Conv(64, (3, 3), padding="SAME"),
        stax.Relu,
        stax.Conv(64, (3, 3), padding="SAME"),
        stax.Relu,
        stax.MaxPool((2, 2), strides=(2, 2), padding="VALID"),
        stax.Conv(128, (3, 3), padding="SAME"),
        stax.Relu,
        stax.Conv(128, (3, 3), padding="SAME"),
        stax.Relu,
        stax.MaxPool((2, 2), strides=(2, 2), padding="VALID"),
        stax.Conv(256, (3, 3), padding="SAME"),
        stax.Relu,
        stax.Conv(256, (3, 3), padding="SAME"),
        stax.Relu,
        stax.Conv(256, (3, 3), padding="SAME"),
        stax.Relu,
        stax.Conv(256, (3, 3), padding="SAME"),
        stax.Relu,
        stax.MaxPool((2, 2), strides=(2, 2), padding="VALID"),
        stax.Conv(512, (3, 3), padding="SAME"),
        stax.Relu,
        stax.Conv(512, (3, 3), padding="SAME"),
        stax.Relu,
        stax.Conv(512, (3, 3), padding="SAME"),
        stax.Relu,
        stax.Conv(512, (3, 3), padding="SAME"),
        stax.Relu,
        stax.MaxPool((2, 2), strides=(2, 2), padding="VALID"),
        stax.Conv(512, (3, 3), padding="SAME"),
        stax.Relu,
        stax.Conv(512, (3, 3), padding="SAME"),
        stax.Relu,
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

    layers.extend([stax.Dense(4096), stax.Relu])

    if dropout_rate:
        layers.append(stax.Dropout(dropout_rate))

    layers.append(stax.Dense(num_classes))
    return stax.serial(*layers)


@ModuleRegistry.register("vgg19-bn")
def vgg19_bn(num_classes: int = 1000, dropout_rate: float | None = 0.5):
    """
    VGG-19 with BatchNorm.
    """
    layers = [
        stax.Conv(64, (3, 3), padding="SAME"),
        stax.BatchNorm(),
        stax.Relu,
        stax.Conv(64, (3, 3), padding="SAME"),
        stax.BatchNorm(),
        stax.Relu,
        stax.MaxPool((2, 2), strides=(2, 2), padding="VALID"),
        stax.Conv(128, (3, 3), padding="SAME"),
        stax.BatchNorm(),
        stax.Relu,
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

    layers.extend([stax.Dense(4096), stax.Relu])

    if dropout_rate:
        layers.append(stax.Dropout(dropout_rate))

    layers.append(stax.Dense(num_classes))
    return stax.serial(*layers)
