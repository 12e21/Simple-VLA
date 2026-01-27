# Simple-VLA

A simple Vision-Language-Action (VLA) model implementation for learning the transformers framework.

## Overview

This project implements a VLA policy that combines:
- **Vision Encoder**: CLIP ViT for image understanding
- **Text Encoder**: BERT for instruction understanding
- **State Encoder**: Neural network for robot state processing
- **Fusion & Action Head**: Multi-modal fusion for action prediction

## Installation

```bash
# Create conda environment
conda create -n simple-vla python=3.10
conda activate simple-vla

# Install ffmpeg (required for video/data processing)
conda install ffmpeg=7 -c conda-forge

# Install Python dependencies
pip install -e .

# Install dev dependencies (includes ruff, pytest, pre-commit)
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Quick Start

### Basic Training

```bash
python train.py
```

This uses the default configuration in `configs/config.yaml`:
- Model: CLIP ViT-Base + BERT-Base
- Dataset: `12e21/so101_pick_box`
- Batch size: 32
- Epochs: 10
- Learning rate: 1e-5

### Debug Mode

Quick debugging with minimal resources:

```bash
python train.py --config-name debug
```

### Override Parameters

Override any configuration parameter from the command line:

```bash
# Change batch size
python train.py training.per_device_train_batch_size=64

# Change learning rate
python train.py training.learning_rate=5e-5

# Change dataset
python train.py data.dataset_id=other/dataset

# Combine multiple overrides
python train.py \
  training.batch_size=64 \
  training.num_train_epochs=20 \
  model.hidden_dim=1024
```

### Hyperparameter Sweeping

Run multiple experiments automatically:

```bash
python train.py --multirun \
  training.learning_rate=1e-4,5e-5,1e-5 \
  model.hidden_dim=512,1024 \
  training.per_device_train_batch_size=32,64
```

## Configuration

All configurations are managed through Hydra in the `configs/` directory:

### File Structure

```
configs/
├── config.yaml      # Default configuration
├── debug.yaml       # Debug preset (1 epoch, small batch)
└── sweep.yaml       # Template for hyperparameter sweeps
```

### Configuration Sections

**Model Configuration** (`model`):
- `vision_model_name`: CLIP model for vision encoding
- `text_model_name`: BERT model for text encoding
- `state_dim`: Dimension of robot state
- `action_dim`: Dimension of action space
- `hidden_dim`: Hidden dimension for fusion layers
- `freeze_encoder`: Whether to freeze encoder weights

**Training Configuration** (`training`):
- Standard Transformers `TrainingArguments` parameters
- Includes output directory, batch size, learning rate, logging, etc.

**Data Configuration** (`data`):
- `dataset_id`: LeRobot dataset identifier
- `split`: Dataset split (train/val/test)
- `num_workers`: DataLoader workers
- `max_samples`: Optional limit on dataset size

## Project Structure

```
simple_vla/
├── config/
│   ├── __init__.py
│   └── utils.py              # Model & training argument builders
├── model/
│   ├── config.py             # SimpleVLAPolicyConfig
│   ├── simple_policy.py      # SimpleVLAPolicyModel
│   └── state_action_processor.py  # Data processors
├── experiment/
│   ├── data_collator.py      # VLACollator
│   └── trainer.py            # VLATrainer
└── tests/
    ├── conftest.py           # Test fixtures
    └── test_simple_policy.py # Model tests

train.py                       # Main training entry point (Hydra)
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=simple_vla

# View available fixtures
pytest --fixtures
```

### Code Formatting

```bash
# Format code with ruff
ruff format .

# Check linting
ruff check .
```

### Pre-commit

Pre-commit hooks automatically run on git commit:
- Ruff formatting
- Ruff linting

## License

MIT License
