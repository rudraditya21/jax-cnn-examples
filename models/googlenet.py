from __future__ import annotations

from jax.example_libraries import stax

from .registry import ModuleRegistry


def _inception_block(
    out_1x1: int,
    red_3x3: int,
    out_3x3: int,
    red_5x5: int,
    out_5x5: int,
    pool_proj: int,
):
    """
    Inception block from GoogLeNet (Szegedy et al., 2014) with 1x1, 3x3, 5x5, and pooled branches.
    """
    branch1 = stax.serial(
        stax.Conv(out_1x1, (1, 1), padding="SAME"),
        stax.Relu,
    )

    branch2 = stax.serial(
        stax.Conv(red_3x3, (1, 1), padding="SAME"),
        stax.Relu,
        stax.Conv(out_3x3, (3, 3), padding="SAME"),
        stax.Relu,
    )

    branch3 = stax.serial(
        stax.Conv(red_5x5, (1, 1), padding="SAME"),
        stax.Relu,
        stax.Conv(out_5x5, (5, 5), padding="SAME"),
        stax.Relu,
    )

    branch4 = stax.serial(
        stax.MaxPool((3, 3), strides=(1, 1), padding="SAME"),
        stax.Conv(pool_proj, (1, 1), padding="SAME"),
        stax.Relu,
    )

    return stax.serial(
        stax.FanOut(4),
        stax.parallel(branch1, branch2, branch3, branch4),
        stax.FanInConcat(axis=-1),
    )


@ModuleRegistry.register("googlenet")
def googlenet(num_classes: int = 1000, dropout_rate: float | None = 0.4):
    """
    GoogLeNet / Inception v1 for ImageNet-sized inputs.
    Note: LRN layers from the original paper are omitted because stax lacks a native operator.
    """
    layers = [
        stax.Conv(64, (7, 7), strides=(2, 2), padding="SAME"),
        stax.Relu,
        stax.MaxPool((3, 3), strides=(2, 2), padding="SAME"),
        stax.Conv(64, (1, 1), padding="SAME"),
        stax.Relu,
        stax.Conv(192, (3, 3), padding="SAME"),
        stax.Relu,
        stax.MaxPool((3, 3), strides=(2, 2), padding="SAME"),
        _inception_block(64, 96, 128, 16, 32, 32),  # 3a
        _inception_block(128, 128, 192, 32, 96, 64),  # 3b
        stax.MaxPool((3, 3), strides=(2, 2), padding="SAME"),
        _inception_block(192, 96, 208, 16, 48, 64),  # 4a
        _inception_block(160, 112, 224, 24, 64, 64),  # 4b
        _inception_block(128, 128, 256, 24, 64, 64),  # 4c
        _inception_block(112, 144, 288, 32, 64, 64),  # 4d
        _inception_block(256, 160, 320, 32, 128, 128),  # 4e
        stax.MaxPool((3, 3), strides=(2, 2), padding="SAME"),
        _inception_block(256, 160, 320, 32, 128, 128),  # 5a
        _inception_block(384, 192, 384, 48, 128, 128),  # 5b
        stax.AvgPool((7, 7), strides=(1, 1), padding="VALID"),
    ]

    if dropout_rate:
        layers.append(stax.Dropout(dropout_rate))

    layers.extend(
        [
            stax.Flatten,
            stax.Dense(num_classes),
        ]
    )

    return stax.serial(*layers)
