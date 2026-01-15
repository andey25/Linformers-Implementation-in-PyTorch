# Linformer Vision Transformer pretraining on CIFAR-100, finetuning on CIFAR-10

## What this notebook contains

This notebook trains a Linformer-based vision transformer on CIFAR-100, then adapts and finetunes the same backbone on CIFAR-10 by replacing the classification head.

It includes dataset setup, model definition, training loops, validation, and final test evaluation for both datasets.

## Model

The model is a patch-based transformer with Linformer attention.

Key settings used in the notebook:
- Patch size: 4
- Embedding dimension: 256
- Layers: 6
- Attention heads: 8
- Dropout: 0.0
- Pretraining head: 100 classes (CIFAR-100)
- Finetune head: 10 classes (CIFAR-10) by overwriting model.mlp_head[1]

## Data

- Pretraining dataset: CIFAR-100 (train, validation split, and test)
- Finetuning dataset: CIFAR-10 (train, validation split, and test)

Both datasets use ToTensor and Normalize with mean (0.5, 0.5, 0.5) and std (0.5, 0.5, 0.5).

## Training and evaluation

- Loss: CrossEntropyLoss
- Optimizer: Adam with lr = 1e-4
- LR scheduler: MultiStepLR with milestones [100, 150] and gamma 0.1
- CIFAR-100 training: 10 epochs
- CIFAR-10 finetuning: 15 epochs
- Validation is tracked each epoch and the best checkpoint by validation loss is saved.

The notebook reports test accuracy for CIFAR-100 and CIFAR-10 at the end of each phase.

## Artifacts

The notebook saves model checkpoints to absolute-style paths like:
- /model_LinformerMain_CIFAR100_{epoch}.pt
- /model_LinformerMain_CIFAR10_{epoch}.pt

Depending on your environment, you may want to change these to a relative directory inside the project.

## Dependencies

Core dependencies used directly in the notebook:
- Python
- PyTorch
- torchvision
- matplotlib

The notebook uses CUDA if available, otherwise it runs on CPU.
