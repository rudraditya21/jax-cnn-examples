from __future__ import annotations

from collections.abc import Sequence

from jax.example_libraries import stax

from .registry import ModuleRegistry


def _make_features(cfg: Sequence[int | str], batch_norm: bool):
    layers: list = []
    for v in cfg:
        if v == "M":
            layers.append(stax.MaxPool((2, 2), strides=(2, 2), padding="VALID"))
        else:
            layers.append(stax.Conv(int(v), (3, 3), padding="SAME"))
            if batch_norm:
                layers.append(stax.BatchNorm())
            layers.append(stax.Relu)
    return layers


def _vgg(
    cfg: Sequence[int | str],
    batch_norm: bool,
    num_classes: int,
    dropout_rate: float | None = 0.5,
):
    layers = _make_features(cfg, batch_norm)
    layers.extend([stax.Flatten, stax.Dense(4096), stax.Relu])

    if dropout_rate:
        layers.append(stax.Dropout(dropout_rate))

    layers.extend([stax.Dense(4096), stax.Relu])

    if dropout_rate:
        layers.append(stax.Dropout(dropout_rate))

    layers.append(stax.Dense(num_classes))
    return stax.serial(*layers)


_CONFIGS: dict[str, Sequence[int | str]] = {
    "vgg11": (64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"),
    "vgg13": (64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"),
    "vgg16": (
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ),
    "vgg19": (
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ),
}


def _register_vgg(name: str, batch_norm: bool):
    @ModuleRegistry.register(name if not batch_norm else f"{name}-bn")
    def factory(num_classes: int = 1000, dropout_rate: float | None = 0.5):
        """
        VGG configuration with optional BatchNorm.
        """
        return _vgg(
            _CONFIGS[name],
            batch_norm=batch_norm,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )

    return factory


vgg11 = _register_vgg("vgg11", batch_norm=False)
vgg11_bn = _register_vgg("vgg11", batch_norm=True)
vgg13 = _register_vgg("vgg13", batch_norm=False)
vgg13_bn = _register_vgg("vgg13", batch_norm=True)
vgg16 = _register_vgg("vgg16", batch_norm=False)
vgg16_bn = _register_vgg("vgg16", batch_norm=True)
vgg19 = _register_vgg("vgg19", batch_norm=False)
vgg19_bn = _register_vgg("vgg19", batch_norm=True)
