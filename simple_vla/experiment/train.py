import PIL.Image
import torch
import torch.optim as optim
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader, Dataset

from simple_vla.model.loss import gaussian_nll_loss
from simple_vla.model.simple_policy import SimpleVLAPolicy
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


def collate_fn(batch):
    """Collate function for DataLoader."""
    images = [item["image"] for item in batch]
    states = torch.stack([item["state"] for item in batch])
    actions = torch.stack([item["action"] for item in batch])
    instructions = [item["instruction"] for item in batch]

    return {"images": images, "states": states, "actions": actions, "instructions": instructions}


def main():
    # 1. Load dataset
    print("Loading dataset...")
    dataset_id = "12e21/so101_pick_box"
    dataset = LeRobotDataset(dataset_id, image_transforms=tensor_to_pil)

    states_min = torch.tensor(dataset.meta.stats["observation.state"]["min"])
    states_max = torch.tensor(dataset.meta.stats["observation.state"]["max"])
    actions_min = torch.tensor(dataset.meta.stats["action"]["min"])
    actions_max = torch.tensor(dataset.meta.stats["action"]["max"])

    # 2. Create DataLoader
    vla_dataset = VLA_Dataset(dataset)
    dataloader = DataLoader(
        vla_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4
    )

    print(f"Dataset size: {len(vla_dataset)}")

    # 3. Initialize model
    print("Initializing model...")
    state_dim = 6
    action_dim = 6

    state_processor = StateProcessor(state_dim, states_min, states_max)
    action_processor = ActionProcessor(action_dim, actions_min, actions_max)

    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    model = SimpleVLAPolicy(
        state_dim=state_dim, action_dim=action_dim, hidden_dim=512, freeze_encoder=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).to(torch.float32)
    print(f"Using device: {device}")

    # 4. Configure optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 5. Training loop
    num_epochs = 10
    print("Starting training...")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            states = state_processor.normalize(batch["states"]).to(device).to(torch.float32)
            actions = action_processor.normalize(batch["actions"]).to(device).to(torch.float32)

            # Forward pass
            action_mean, action_log_std = model(batch["images"], batch["instructions"], states)

            # Compute loss
            loss = gaussian_nll_loss(action_mean, action_log_std, actions)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}, Loss: {loss.item():.4f}")

        # Epoch statistics
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                f"checkpoint_epoch_{epoch + 1}.pt",
            )
            print(f"  Saved checkpoint_epoch_{epoch + 1}.pt")

    print("Training completed!")


if __name__ == "__main__":
    main()
