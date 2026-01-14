"""Custom Trainer for VLA policy with Gaussian NLL loss."""

from typing import Any, Dict, Optional, Tuple, Union

import torch
from transformers import Trainer

from simple_vla.model.loss import gaussian_nll_loss


class VLATrainer(Trainer):
    """
    Custom Trainer for VLA policy training.

    Overrides compute_loss to use gaussian_nll_loss instead of standard cross-entropy.
    The model outputs (action_mean, action_log_std) instead of direct loss.
    """

    def compute_loss(
        self,
        model: Any,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """
        Compute Gaussian NLL loss for VLA policy training.

        Args:
            model: SimpleVLAPolicyModel
            inputs: Dict with keys:
                - images: List[PIL.Image]
                - instructions: List[str]
                - states: torch.Tensor (normalized)
                - actions: torch.Tensor (normalized, target)
            return_outputs: If True, return (loss, (mean, log_std))
            num_items_in_batch: Optional batch size (unused, for compatibility)

        Returns:
            loss tensor, or (loss, (action_mean, action_log_std)) if return_outputs
        """
        # Extract inputs
        images = inputs.pop("images")
        instructions = inputs.pop("instructions")
        states = inputs.pop("states")
        actions = inputs.pop("actions")

        # Forward pass: model returns (action_mean, action_log_std)
        action_mean, action_log_std = model(images, instructions, states)

        # Compute Gaussian negative log-likelihood loss
        loss = gaussian_nll_loss(action_mean, action_log_std, actions)

        return (loss, (action_mean, action_log_std)) if return_outputs else loss

    def prediction_step(
        self,
        model: Any,
        inputs: Dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Override to handle custom model output format.

        Returns:
            (loss, logits, labels) where:
            - loss: optional tensor
            - logits: action_mean concatenated with action_log_std
            - labels: target actions
        """
        # Extract inputs
        images = inputs.pop("images")
        instructions = inputs.pop("instructions")
        states = inputs.pop("states")
        actions = inputs.pop("actions")

        # Forward pass
        with torch.no_grad():
            action_mean, action_log_std = model(images, instructions, states)
            loss = gaussian_nll_loss(action_mean, action_log_std, actions)

        # Concatenate mean and log_std as "logits" for metrics
        logits = torch.cat([action_mean, action_log_std], dim=-1)

        return (loss, logits, actions)
