from __future__ import annotations

from jax.example_libraries import stax

from .registry import ModuleRegistry


def _fire(squeeze_ch: int, expand1x1: int, expand3x3: int):
    """
    Fire module: squeeze 1x1 conv followed by parallel 1x1 and 3x3 expand layers.

    Args:
        squeeze_ch: Channels for squeeze layer.
        expand1x1: Channels for 1x1 expand branch.
        expand3x3: Channels for 3x3 expand branch.

    Returns:
        `(init_fn, apply_fn)` pair composing a fire block.
    """
    return stax.serial(
        stax.Conv(squeeze_ch, (1, 1), padding="SAME"),
        stax.Relu,
        stax.FanOut(2),
        stax.parallel(
            stax.serial(
                stax.Conv(expand1x1, (1, 1), padding="SAME"),
                stax.Relu,
            ),
            stax.serial(
                stax.Conv(expand3x3, (3, 3), padding="SAME"),
                stax.Relu,
            ),
        ),
        stax.FanInConcat(axis=-1),
    )


@ModuleRegistry.register("squeezenet1.0")
def squeezenet_1_0(num_classes: int = 1000):
    """
    SqueezeNet v1.0.

    Args:
        num_classes: Number of output classes.

    Returns:
        `(init_fn, apply_fn)` tuple for SqueezeNet v1.0.
    """
    return stax.serial(
        stax.Conv(96, (7, 7), strides=(2, 2), padding="SAME"),
        stax.Relu,
        stax.MaxPool((3, 3), strides=(2, 2), padding="SAME"),
        _fire(16, 64, 64),
        _fire(16, 64, 64),
        _fire(32, 128, 128),
        stax.MaxPool((3, 3), strides=(2, 2), padding="SAME"),
        _fire(32, 128, 128),
        _fire(48, 192, 192),
        _fire(48, 192, 192),
        _fire(64, 256, 256),
        stax.MaxPool((3, 3), strides=(2, 2), padding="SAME"),
        _fire(64, 256, 256),
        stax.Conv(num_classes, (1, 1), padding="SAME"),
        stax.Relu,
        stax.AvgPool((13, 13), strides=(1, 1), padding="VALID"),
        stax.Flatten,
    )


@ModuleRegistry.register("squeezenet1.1")
def squeezenet_1_1(num_classes: int = 1000):
    """
    SqueezeNet v1.1 (more efficient stem).

    Args:
        num_classes: Number of output classes.

    Returns:
        `(init_fn, apply_fn)` tuple for SqueezeNet v1.1.
    """
    return stax.serial(
        stax.Conv(64, (3, 3), strides=(2, 2), padding="SAME"),
        stax.Relu,
        stax.MaxPool((3, 3), strides=(2, 2), padding="SAME"),
        _fire(16, 64, 64),
        _fire(16, 64, 64),
        stax.MaxPool((3, 3), strides=(2, 2), padding="SAME"),
        _fire(32, 128, 128),
        _fire(32, 128, 128),
        stax.MaxPool((3, 3), strides=(2, 2), padding="SAME"),
        _fire(48, 192, 192),
        _fire(48, 192, 192),
        _fire(64, 256, 256),
        _fire(64, 256, 256),
        stax.Conv(num_classes, (1, 1), padding="SAME"),
        stax.Relu,
        stax.AvgPool((13, 13), strides=(1, 1), padding="VALID"),
        stax.Flatten,
    )
