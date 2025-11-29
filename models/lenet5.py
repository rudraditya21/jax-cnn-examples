from __future__ import annotations

from jax.example_libraries import stax

from .registry import ModuleRegistry


@ModuleRegistry.register("lenet5")
def lenet5(num_classes: int = 10):
    """
    Canonical LeNet-5 for 32x32 grayscale inputs.
    Conv(6,5x5) -> AvgPool -> Conv(16,5x5) -> AvgPool -> Conv(120,5x5) -> Flatten -> Dense(84) -> Dense(num_classes)
    """
    return stax.serial(
        stax.Conv(6, (5, 5), padding="VALID"),
        stax.Tanh,
        stax.AvgPool((2, 2), strides=(2, 2), padding="VALID"),
        stax.Conv(16, (5, 5), padding="VALID"),
        stax.Tanh,
        stax.AvgPool((2, 2), strides=(2, 2), padding="VALID"),
        stax.Conv(120, (5, 5), padding="VALID"),
        stax.Tanh,
        stax.Flatten,
        stax.Dense(84),
        stax.Tanh,
        stax.Dense(num_classes),
    )
