from __future__ import annotations

from jax.example_libraries import stax

from .registry import ModuleRegistry


@ModuleRegistry.register("zfnet")
def zfnet(num_classes: int = 1000, dropout_rate: float | None = 0.5):
    """
    ZFNet (Zeiler & Fergus, 2013) for ImageNet-sized inputs.
    Note: LRN layers are omitted because stax lacks a native operator; rest mirrors the original layout.
    """
    layers = [
        stax.Conv(96, (7, 7), strides=(2, 2), padding="SAME"),
        stax.Relu,
        stax.MaxPool((3, 3), strides=(2, 2), padding="VALID"),
        stax.Conv(256, (5, 5), strides=(2, 2), padding="SAME"),
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
