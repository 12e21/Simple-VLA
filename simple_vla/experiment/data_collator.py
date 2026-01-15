"""Data collator for VLA training with Transformers Trainer."""

from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from simple_vla.model.state_action_processor import ActionProcessor, StateProcessor


@dataclass
class VLACollator:
    """
    Custom data collator for VLA multi-modal training.

    Handles:
    - PIL Images (kept as list for encode_image)
    - Text instructions (kept as list for encode_text)
    - State/Action tensor normalization
    """

    state_processor: StateProcessor
    action_processor: ActionProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate batch of features and normalize states/actions.

        Args:
            features: List of dicts with keys:
                - image: PIL.Image
                - state: torch.Tensor
                - action: torch.Tensor
                - instruction: str

        Returns:
            Dict with keys:
                - images: List[PIL.Image]
                - instructions: List[str]
                - states: torch.Tensor (normalized)
                - actions: torch.Tensor (normalized)
        """
        images = [item["image"] for item in features]
        instructions = [item["instruction"] for item in features]
        states = torch.stack([item["state"] for item in features])
        actions = torch.stack([item["action"] for item in features])

        # Normalize states and actions
        normalized_states = self.state_processor.normalize(states)
        normalized_actions = self.action_processor.normalize(actions)

        # Ensure float32 dtype for model compatibility
        normalized_states = normalized_states.to(torch.float32)
        normalized_actions = normalized_actions.to(torch.float32)

        return {
            "images": images,
            "instructions": instructions,
            "states": normalized_states,
            "actions": normalized_actions,
        }
