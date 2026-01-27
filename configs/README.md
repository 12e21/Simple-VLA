# Hydra Configuration System

This project uses Hydra for flexible experiment configuration.

## Configuration Files

- **`config.yaml`**: Default configuration for all parameters
- **`debug.yaml`**: Quick debugging preset (1 epoch, small batch)
- **`sweep.yaml`**: Template for hyperparameter sweeping

## Usage Examples

### Basic Training
```bash
python train.py
```

### Debug Mode
```bash
python train.py --config-name debug
```

### Override Single Parameters
```bash
python train.py training.learning_rate=5e-5
python train.py model.hidden_dim=1024
python train.py data.dataset_id=other/dataset
```

### Hyperparameter Sweep
```bash
python train.py --multirun \
  training.learning_rate=1e-4,5e-5,1e-5 \
  model.hidden_dim=512,1024
```

### Override Multiple Parameters
```bash
python train.py \
  training.batch_size=64 \
  training.num_train_epochs=20 \
  model.freeze_encoder=false
```

## Configuration Structure

```yaml
model:
  vision_model_name: "openai/clip-vit-base-patch32"
  text_model_name: "bert-base-uncased"
  state_dim: 6
  action_dim: 6
  hidden_dim: 512
  freeze_encoder: true

training:
  output_dir: "./outputs/simple_vla"
  num_train_epochs: 10
  per_device_train_batch_size: 32
  learning_rate: 1.0e-5
  # ... (TrainingArguments parameters)

data:
  dataset_id: "12e21/so101_pick_box"
  split: "train"
  num_workers: 4
  pin_memory: true

seed: 42
```

## Output

Hydra automatically creates organized output directories:
```
outputs/
├── YYYY-MM-DD/
│   ├── HH-MM-SS/          # Each run gets timestamped
│   │   ├── .hydra/        # Hydra config logs
│   │   ├── checkpoints/   # Model checkpoints
│   │   └── ...
```
