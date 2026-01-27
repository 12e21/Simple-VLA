"""Training script for VLA policy using Transformers Trainer with Hydra configuration."""

import PIL.Image
import torch
from hydra import main
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import Trainer

from simple_vla.config import build_model_from_cfg, build_training_arguments_from_cfg
from simple_vla.experiment.data_collator import VLACollator
from simple_vla.experiment.trainer import VLATrainer
from simple_vla.model.state_action_processor import ActionProcessor, StateProcessor


def tensor_to_pil(img_tensor):
    """Convert tensor to PIL Image."""
    img_tensor = img_tensor * 255.0
    img_tensor = img_tensor.clamp(0, 255).byte()
    img_tensor = img_tensor.permute(1, 2, 0)
    img_np = img_tensor.cpu().numpy()
    return PIL.Image.fromarray(img_np, mode="RGB")


class VLA_Dataset(Dataset):
    """Simple wrapper for LeRobotDataset."""

    def __init__(self, lerobot_dataset, max_samples=None):
        self.dataset = lerobot_dataset
        self.max_samples = max_samples

    def __len__(self):
        if self.max_samples is not None:
            return min(self.max_samples, len(self.dataset))
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        return {
            "image": sample["observation.images.front"],
            "state": sample["observation.state"],
            "action": sample["action"],
            "instruction": sample["task"],
        }


@main(version_base="1.2", config_path="./configs", config_name="config")
def train(cfg: DictConfig) -> None:
    """Main training function with Hydra config.

    Args:
        cfg: Hydra configuration object
    """
    # Set random seed
    torch.manual_seed(cfg.seed)

    # 1. Load dataset
    print("Loading dataset...")
    dataset = LeRobotDataset(cfg.data.dataset_id, image_transforms=tensor_to_pil)

    states_min = torch.tensor(dataset.meta.stats["observation.state"]["min"])
    states_max = torch.tensor(dataset.meta.stats["observation.state"]["max"])
    actions_min = torch.tensor(dataset.meta.stats["action"]["min"])
    actions_max = torch.tensor(dataset.meta.stats["action"]["max"])

    # 2. Create dataset wrapper
    max_samples = cfg.data.get("max_samples", None)
    vla_dataset = VLA_Dataset(dataset, max_samples=max_samples)
    print(f"Dataset size: {len(vla_dataset)}")

    # 3. Initialize processors
    print("Initializing processors...")
    state_processor = StateProcessor(cfg.model.state_dim, states_min, states_max)
    action_processor = ActionProcessor(cfg.model.action_dim, actions_min, actions_max)

    print(f"State dim: {cfg.model.state_dim}, Action dim: {cfg.model.action_dim}")

    # 4. Initialize model from config
    print("Initializing model...")
    model = build_model_from_cfg(cfg)

    # 5. Create data collator with processors
    data_collator = VLACollator(
        state_processor=state_processor,
        action_processor=action_processor,
    )

    # 6. Configure training arguments
    training_args = build_training_arguments_from_cfg(cfg)

    # 7. Initialize Trainer
    print("Initializing trainer...")
    trainer: Trainer = VLATrainer(
        model=model,
        args=training_args,
        train_dataset=vla_dataset,
        data_collator=data_collator,
    )

    # 8. Train
    print("Starting training...")
    trainer.train()

    # 9. Save final model
    print("Saving model...")
    trainer.save_model(cfg.training.output_dir)
    print("Training completed!")


if __name__ == "__main__":
    main()
