"""Pytest configuration and fixtures for testing."""

from collections.abc import Generator

import pytest
import torch


@pytest.fixture
def sample_image() -> torch.Tensor:
    """Generate a sample RGB image tensor matching observation.images.front format.

    Returns:
        torch.Tensor: Image tensor of shape [3, 480, 640] (C, H, W)
    """
    return torch.randn(3, 480, 640)


@pytest.fixture
def sample_action() -> torch.Tensor:
    """Generate a sample action vector.

    Returns:
        torch.Tensor: Action vector of shape [6]
    """
    return torch.randn(6)


@pytest.fixture
def sample_state() -> torch.Tensor:
    """Generate a sample state vector.

    Returns:
        torch.Tensor: State vector of shape [6]
    """
    return torch.randn(6)


@pytest.fixture
def sample_task() -> str:
    """Generate a sample task description.

    Returns:
        str: Task description
    """
    return "put the little box into box"


@pytest.fixture
def sample_dataset_entry(
    sample_image: torch.Tensor,
    sample_action: torch.Tensor,
    sample_state: torch.Tensor,
    sample_task: str,
) -> dict[str, torch.Tensor | str | int]:
    """Generate a single dataset entry matching the real dataset format.

    Args:
        sample_image: Image tensor fixture
        sample_action: Action vector fixture
        sample_state: State vector fixture
        sample_task: Task description fixture

    Returns:
        dict: A single dataset entry with all required keys
    """
    return {
        "observation.images.front": sample_image,
        "action": sample_action,
        "observation.state": sample_state,
        "timestamp": 0.0,
        "frame_index": 0,
        "episode_index": 0,
        "index": 0,
        "task_index": 0,
        "task": sample_task,
    }


@pytest.fixture
def sample_batch_size() -> int:
    """Standard batch size for testing.

    Returns:
        int: Batch size of 8
    """
    return 8


@pytest.fixture
def sample_batch(
    sample_batch_size: int,
    sample_image: torch.Tensor,
    sample_action: torch.Tensor,
    sample_state: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Generate a batch of data for testing.

    Args:
        sample_batch_size: Number of samples in batch
        sample_image: Single image tensor fixture
        sample_action: Single action vector fixture
        sample_state: Single state vector fixture

    Returns:
        dict: Batched data with tensors shaped for batch processing
    """
    return {
        "observation.images.front": torch.randn(sample_batch_size, 3, 480, 640),
        "action": torch.randn(sample_batch_size, 6),
        "observation.state": torch.randn(sample_batch_size, 6),
    }


@pytest.fixture
def sample_sequence_length() -> int:
    """Standard sequence length for temporal models.

    Returns:
        int: Sequence length of 10
    """
    return 10


@pytest.fixture
def sample_temporal_batch(
    sample_batch_size: int,
    sample_sequence_length: int,
) -> dict[str, torch.Tensor]:
    """Generate a temporal batch for sequence model testing.

    Args:
        sample_batch_size: Number of sequences in batch
        sample_sequence_length: Length of each sequence

    Returns:
        dict: Batched temporal data with sequence dimension
    """
    return {
        "observation.images.front": torch.randn(
            sample_batch_size, sample_sequence_length, 3, 480, 640
        ),
        "action": torch.randn(sample_batch_size, sample_sequence_length, 6),
        "observation.state": torch.randn(sample_batch_size, sample_sequence_length, 6),
    }


class FakeDataset:
    """Fake dataset class for testing that mimics LeRobotDataset interface."""

    def __init__(self, size: int = 100):
        """Initialize fake dataset with specified size.

        Args:
            size: Number of samples in the dataset
        """
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        if idx >= self.size:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.size}")
        return {
            "observation.images.front": torch.randn(3, 480, 640),
            "action": torch.randn(6),
            "observation.state": torch.randn(6),
            "timestamp": float(idx) * 0.1,
            "frame_index": idx % 100,
            "episode_index": idx // 100,
            "index": idx,
            "task_index": 0,
            "task": "put the little box into box",
        }


@pytest.fixture
def fake_dataset() -> Generator[FakeDataset, None, None]:
    """Provide a fake dataset instance for testing.

    Yields:
        FakeDataset: Fake dataset with 100 samples
    """
    dataset = FakeDataset(size=100)
    yield dataset


@pytest.fixture
def small_dataset() -> Generator[FakeDataset, None, None]:
    """Provide a small fake dataset for quick tests.

    Yields:
        FakeDataset: Fake dataset with 10 samples
    """
    dataset = FakeDataset(size=10)
    yield dataset


# Fixtures for SimpleVLAPolicy tests


@pytest.fixture
def single_image() -> torch.Tensor:
    """Single image for testing VLA policy.

    Returns:
        torch.Tensor: Image tensor of shape [1, 3, 224, 224] for CLIP input
    """
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def dummy_image_batch() -> torch.Tensor:
    """Batch of images for testing VLA policy.

    Returns:
        torch.Tensor: Image batch of shape [batch_size, 3, 224, 224]
    """
    return torch.randn(8, 3, 224, 224)


@pytest.fixture
def single_text() -> list[str]:
    """Single text instruction for testing VLA policy.

    Returns:
        list[str]: List with one task description
    """
    return ["pick up the red block"]


@pytest.fixture
def dummy_text_batch() -> list[str]:
    """Batch of text instructions for testing VLA policy.

    Returns:
        list[str]: List of task descriptions
    """
    return [
        "pick up the red block",
        "move to the left",
        "place the object on the table",
        "grasp the blue cup",
        "push the box forward",
        "rotate the handle",
        "open the drawer",
        "close the door",
    ]
