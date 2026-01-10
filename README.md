# JAX CNN Examples

Minimal JAX CNN model zoo with a quick training sanity check.

## Usage

List models and datasets:

```bash
python3 main.py list --section all
```

Run a short training sanity check:

```bash
python3 main.py train --dataset cifar10 --model resnet18 --epochs 1 --max-steps 50
```

## Implemented Models

| Model | Model | Model | Model |
| --- | --- | --- | --- |
| alexnet | alexnet-lite | convnext-s | convnext-t |
| densenet121 | densenet161 | densenet169 | densenet201 |
| efficientnet-b0 | efficientnet-b1 | efficientnet-b2 | efficientnet-b3 |
| efficientnet-b4 | efficientnet-b5 | efficientnet-b6 | efficientnet-b7 |
| googlenet | lenet1 | lenet4 | lenet5 |
| mobilenetv1-0.25 | mobilenetv1-0.5 | mobilenetv1-0.75 | mobilenetv1-1.0 |
| mobilenetv2-0.35 | mobilenetv2-0.5 | mobilenetv2-0.75 | mobilenetv2-1.0 |
| nfnet-f0 | nfnet-f1 | regnetx-400mf | regnetx-800mf |
| repvgg-a0 | repvgg-b1 | resnet101 | resnet152 |
| resnet18 | resnet34 | resnet50 | resnext101-32x4d |
| resnext101-64x4d | resnext50-32x4d | shufflenetv2-0.5 | shufflenetv2-1.0 |
| shufflenetv2-1.5 | shufflenetv2-2.0 | squeezenet1.0 | squeezenet1.1 |
| vgg11 | vgg11-bn | vgg13 | vgg13-bn |
| vgg16 | vgg16-bn | vgg19 | vgg19-bn |
| wideresnet16-8 | wideresnet28-10 | wideresnet40-2 | zfnet |

## Implemented Datasets

| Dataset | Dataset | Dataset |
| --- | --- | --- |
| cifar10 | cifar100 | country211 |
| emnist | eurosat | fashion_mnist |
| fer2013 | gtsrb | kmnist |
| mnist | pcam | qmnist |
| sun397 |  |  |
