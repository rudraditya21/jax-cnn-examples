import argparse

parser = argparse.ArgumentParser("Train CNN Architecture")

parser.add_argument("--dataset", type=str.lower, required=True, choices=[])
parser.add_argument("--data-dir", type=str, default="./data")
