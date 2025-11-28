import argparse

from datasets.registry import DatasetRegistry

parser = argparse.ArgumentParser(description="Train CNN Architecture")

parser.add_argument(
    "--dataset",
    type=str.lower,
    required=True,
    choices=DatasetRegistry.list_datasets(),
)
parser.add_argument("--data-dir", type=str, default="./data")

args = parser.parse_args()
