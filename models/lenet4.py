from __future__ import annotations

from jax.example_libraries import stax

from .registry import ModuleRegistry


@ModuleRegistry.register("lenet4")
def lenet4(num_classes: int = 10):
    """
    LeNet-4 variant for 32x32 inputs.
    Conv(6,5x5) -> AvgPool -> Conv(16,5x5) -> AvgPool -> Flatten -> Dense(120) -> Dense(84) -> Dense(num_classes)
    """
    return stax.serial(
        stax.Conv(6, (5, 5), padding="VALID"),
        stax.Tanh,
        stax.AvgPool((2, 2), strides=(2, 2), padding="VALID"),
        stax.Conv(16, (5, 5), padding="VALID"),
        stax.Tanh,
        stax.AvgPool((2, 2), strides=(2, 2), padding="VALID"),
        stax.Flatten,
        stax.Dense(120),
        stax.Tanh,
        stax.Dense(84),
        stax.Tanh,
        stax.Dense(num_classes),
    )
