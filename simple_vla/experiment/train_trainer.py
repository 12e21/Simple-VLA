"""Training script for VLA policy using Transformers Trainer."""

import PIL.Image
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import Dataset
from transformers import TrainingArguments

from simple_vla.experiment.data_collator import VLACollator
from simple_vla.experiment.trainer import VLATrainer
from simple_vla.model.config import SimpleVLAPolicyConfig
from simple_vla.model.simple_policy import SimpleVLAPolicyModel
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

    def __init__(self, lerobot_dataset):
        self.dataset = lerobot_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        return {
            "image": sample["observation.images.front"],
            "state": sample["observation.state"],
            "action": sample["action"],
            "instruction": sample["task"],
        }


def main():
    # 1. Load dataset
    print("Loading dataset...")
    dataset_id = "12e21/so101_pick_box"
    dataset = LeRobotDataset(dataset_id, image_transforms=tensor_to_pil)

    states_min = torch.tensor(dataset.meta.stats["observation.state"]["min"])
    states_max = torch.tensor(dataset.meta.stats["observation.state"]["max"])
    actions_min = torch.tensor(dataset.meta.stats["action"]["min"])
    actions_max = torch.tensor(dataset.meta.stats["action"]["max"])

    # 2. Create dataset wrapper
    vla_dataset = VLA_Dataset(dataset)
    print(f"Dataset size: {len(vla_dataset)}")

    # 3. Initialize processors
    print("Initializing processors...")
    state_dim = 6
    action_dim = 6

    state_processor = StateProcessor(state_dim, states_min, states_max)
    action_processor = ActionProcessor(action_dim, actions_min, actions_max)

    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    # 4. Initialize model from config
    print("Initializing model...")
    config = SimpleVLAPolicyConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=512,
        freeze_encoder=True,
    )
    model = SimpleVLAPolicyModel(config)

    # 5. Create data collator with processors
    data_collator = VLACollator(
        state_processor=state_processor,
        action_processor=action_processor,
    )

    # 6. Configure training arguments
    training_args = TrainingArguments(
        output_dir="./outputs/simple_vla",
        num_train_epochs=10,
        per_device_train_batch_size=32,
        learning_rate=1e-5,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        remove_unused_columns=False,  # Keep custom columns (images, instructions)
        report_to=["wandb"],
    )

    # 7. Initialize Trainer
    print("Initializing trainer...")
    trainer = VLATrainer(
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
    trainer.save_model("./outputs/simple_vla/")
    print("Training completed!")


if __name__ == "__main__":
    main()
