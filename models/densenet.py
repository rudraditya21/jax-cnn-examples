from __future__ import annotations

from collections.abc import Sequence

from jax.example_libraries import stax

from .registry import ModuleRegistry


def _dense_layer(growth_rate: int):
    """
    Bottleneck dense layer: BN-ReLU-1x1 -> BN-ReLU-3x3, concatenated with input.
    """
    bottleneck_channels = 4 * growth_rate

    branch = stax.serial(
        stax.BatchNorm(),
        stax.Relu,
        stax.Conv(bottleneck_channels, (1, 1), padding="SAME"),
        stax.BatchNorm(),
        stax.Relu,
        stax.Conv(growth_rate, (3, 3), padding="SAME"),
    )

    return stax.serial(
        stax.FanOut(2),
        stax.parallel(branch, stax.Identity),
        stax.FanInConcat(axis=-1),
    )


def _dense_block(num_layers: int, growth_rate: int):
    return stax.serial(*(_dense_layer(growth_rate) for _ in range(num_layers)))


def _transition(out_channels: int):
    return stax.serial(
        stax.BatchNorm(),
        stax.Relu,
        stax.Conv(out_channels, (1, 1), padding="SAME"),
        stax.AvgPool((2, 2), strides=(2, 2), padding="SAME"),
    )


def _densenet(
    block_layers: Sequence[int],
    growth_rate: int,
    num_classes: int,
    compression: float = 0.5,
    initial_channels: int = 64,
):
    layers: list = [
        stax.Conv(initial_channels, (7, 7), strides=(2, 2), padding="SAME"),
        stax.BatchNorm(),
        stax.Relu,
        stax.MaxPool((3, 3), strides=(2, 2), padding="SAME"),
    ]

    channels = initial_channels
    last_block = len(block_layers) - 1

    for i, block_size in enumerate(block_layers):
        layers.append(_dense_block(block_size, growth_rate))
        channels += block_size * growth_rate

        if i != last_block:
            out_channels = int(channels * compression)
            layers.append(_transition(out_channels))
            channels = out_channels

    layers.extend(
        [
            stax.BatchNorm(),
            stax.Relu,
            stax.AvgPool((7, 7), strides=(1, 1), padding="VALID"),
            stax.Flatten,
            stax.Dense(num_classes),
        ]
    )

    return stax.serial(*layers)


@ModuleRegistry.register("densenet121")
def densenet121(num_classes: int = 1000):
    """
    DenseNet-121 (Huang et al., 2017) with growth rate 32.
    """
    return _densenet((6, 12, 24, 16), growth_rate=32, num_classes=num_classes)


@ModuleRegistry.register("densenet169")
def densenet169(num_classes: int = 1000):
    """
    DenseNet-169 (Huang et al., 2017) with growth rate 32.
    """
    return _densenet((6, 12, 32, 32), growth_rate=32, num_classes=num_classes)


@ModuleRegistry.register("densenet201")
def densenet201(num_classes: int = 1000):
    """
    DenseNet-201 (Huang et al., 2017) with growth rate 32.
    """
    return _densenet((6, 12, 48, 32), growth_rate=32, num_classes=num_classes)


@ModuleRegistry.register("densenet161")
def densenet161(num_classes: int = 1000):
    """
    DenseNet-161 (Huang et al., 2017) with growth rate 48 and wider stem.
    """
    return _densenet(
        (6, 12, 36, 24),
        growth_rate=48,
        num_classes=num_classes,
        initial_channels=96,
    )
