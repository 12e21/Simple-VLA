"""VLA Policy Inference Pipeline.

This module provides a custom Hugging Face Transformers Pipeline for Vision-Language-Action (VLA)
model inference. The pipeline accepts numpy arrays as input for easy deployment on real robots.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers import Pipeline

from simple_vla.model import ActionProcessor, SimpleVLAPolicyModel, StateProcessor


class VLAPolicyPipeline(Pipeline):
    """Pipeline for Vision-Language-Action (VLA) policy inference.

    This pipeline accepts multimodal inputs (images, text instructions, and robot states)
    and outputs action predictions for robot control.

    Args:
        model: The VLA model (SimpleVLAPolicyModel)
        state_processor: StateProcessor for normalizing/denormalizing states
        action_processor: ActionProcessor for normalizing/denormalizing actions
        **kwargs: Additional arguments passed to Pipeline base class

    Example:
        >>> from simple_vla.model import SimpleVLAPolicyModel, StateProcessor, ActionProcessor
        >>> from simple_vla.eval.pipeline_infer import VLAPolicyPipeline
        >>> model = SimpleVLAPolicyModel.from_pretrained("path/to/checkpoint")
        >>> state_proc = StateProcessor.load("path/to/state_processor.pt")
        >>> action_proc = ActionProcessor.load("path/to/action_processor.pt")
        >>> pipe = VLAPolicyPipeline(
        ...     model=model, state_processor=state_proc, action_processor=action_proc
        ... )
        >>>
        >>> # Prepare inputs (numpy arrays)
        >>> image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        >>> state = np.random.randn(14)  # Robot state
        >>> text = "Pick up the red block"
        >>>
        >>> # Run inference
        >>> result = pipe(images=image, text=text, states=state)
        >>> print(result['action'])  # Predicted action
    """

    def __init__(
        self,
        model: SimpleVLAPolicyModel,
        state_processor: Optional[StateProcessor] = None,
        action_processor: Optional[ActionProcessor] = None,
        **kwargs,
    ):
        """Initialize the VLA policy pipeline.

        Args:
            model: The VLA model instance
            state_processor: Optional state processor for normalization
            action_processor: Optional action processor for denormalization
            **kwargs: Additional arguments for base Pipeline class
        """
        super().__init__(model=model, **kwargs)
        self.state_processor = state_processor
        self.action_processor = action_processor

        # Store processors from model if available
        if hasattr(model, "model") and hasattr(model.model, "image_processor"):
            self.image_processor = model.model.image_processor
        else:
            raise ValueError("Model must have an image_processor attribute")

        if hasattr(model, "model") and hasattr(model.model, "text_tokenizer"):
            self.text_tokenizer = model.model.text_tokenizer
        else:
            raise ValueError("Model must have a text_tokenizer attribute")

    def _sanitize_parameters(
        self, return_std: bool = False, deterministic: bool = True, **kwargs
    ) -> tuple:
        """Sanitize parameters for preprocessing and postprocessing.

        Args:
            return_std: Whether to return action standard deviation
            deterministic: Whether to return deterministic actions (mean only)
            **kwargs: Additional parameters

        Returns:
            Tuple of (preprocess_params, forward_params, postprocess_params)
        """
        preprocess_kwargs = {}
        forward_kwargs = {}
        postprocess_kwargs = {"return_std": return_std, "deterministic": deterministic}

        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, torch.Tensor]:
        """Preprocess inputs for the model.

        Args:
            inputs: Dictionary containing:
                - images: np.ndarray or list of np.ndarray (H, W, C) in RGB format
                - text: str or list of str
                - states: np.ndarray or list of np.ndarray
            **kwargs: Additional preprocessing parameters

        Returns:
            Dictionary with preprocessed tensors ready for model input

        Raises:
            ValueError: If required keys are missing or input shapes are invalid
        """
        if not isinstance(inputs, dict):
            raise ValueError("Inputs must be a dictionary with 'images', 'text', and 'states' keys")

        # Extract inputs
        images = inputs.get("images")
        text = inputs.get("text")
        states = inputs.get("states")

        if images is None or text is None or states is None:
            raise ValueError("Inputs must contain 'images', 'text', and 'states' keys")

        # Process images - convert numpy arrays to PIL Images
        if isinstance(images, np.ndarray):
            if images.ndim == 3:  # Single image (H, W, C)
                images = [Image.fromarray(images.astype(np.uint8))]
            elif images.ndim == 4:  # Batch of images (B, H, W, C)
                images = [Image.fromarray(img.astype(np.uint8)) for img in images]
            else:
                raise ValueError(
                    f"Images must be 3D (H, W, C) or 4D (B, H, W, C), got shape {images.shape}"
                )
        elif isinstance(images, list):
            # List of numpy arrays or PIL Images
            processed_images = []
            for img in images:
                if isinstance(img, np.ndarray):
                    if img.ndim == 3:
                        processed_images.append(Image.fromarray(img.astype(np.uint8)))
                    else:
                        raise ValueError(f"Image array must be 3D (H, W, C), got shape {img.shape}")
                elif isinstance(img, Image.Image):
                    processed_images.append(img)
                else:
                    raise ValueError(f"Unsupported image type: {type(img)}")
            images = processed_images
        else:
            raise ValueError(f"Images must be np.ndarray or list, got {type(images)}")

        # Process states - convert to tensor and normalize
        if isinstance(states, np.ndarray):
            if states.ndim == 1:  # Single state
                states = torch.from_numpy(states.astype(np.float32)).unsqueeze(0)
            elif states.ndim == 2:  # Batch of states (B, state_dim)
                states = torch.from_numpy(states.astype(np.float32))
            else:
                raise ValueError(
                    f"States must be 1D (state_dim,) or 2D (B, state_dim), got shape {states.shape}"
                )
        elif isinstance(states, list):
            states = torch.stack(
                [
                    torch.from_numpy(s.astype(np.float32)) if isinstance(s, np.ndarray) else s
                    for s in states
                ]
            )
        elif isinstance(states, torch.Tensor):
            pass  # Already a tensor
        else:
            raise ValueError(
                f"States must be np.ndarray, list, or torch.Tensor, got {type(states)}"
            )

        # Normalize states if processor is available
        if self.state_processor is not None:
            original_device = states.device
            states = self.state_processor.normalize(states)
            # Move back to original device for consistency
            states = states.to(original_device)

        # Process text - tokenize
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list):
            raise ValueError(f"Text must be str or list of str, got {type(text)}")

        # Move all tensors to the correct device
        device = self.device
        states = states.to(device)

        # Encode images using model's image processor
        with torch.no_grad():
            image_inputs = self.image_processor(images, return_tensors="pt").to(device)
            vision_outputs = self.model.model.vision_encoder(**image_inputs)
            vision_embeddings = vision_outputs.pooler_output

        # Tokenize text
        text_inputs = self.text_tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        with torch.no_grad():
            text_outputs = self.model.model.text_encoder(**text_inputs)
            text_embeddings = text_outputs.pooler_output

        # Prepare model inputs
        model_inputs = {
            "vision_embeddings": vision_embeddings,
            "text_embeddings": text_embeddings,
            "states": states,
        }

        return model_inputs

    def _forward(self, model_inputs: Dict[str, torch.Tensor], **kwargs) -> tuple:
        """Run model inference.

        Args:
            model_inputs: Dictionary with preprocessed inputs
            **kwargs: Additional forward parameters

        Returns:
            Tuple of (action_mean, action_log_std) tensors
        """
        # Extract embeddings and states
        vision_embeddings = model_inputs["vision_embeddings"]
        text_embeddings = model_inputs["text_embeddings"]
        states = model_inputs["states"]

        # Run state encoder
        state_embeddings = self.model.model.state_encoder(states)

        # Fuse embeddings
        fused_embeddings = self.model.model.fuse_layers(
            torch.cat((vision_embeddings, text_embeddings, state_embeddings), dim=-1)
        )

        # Get action predictions
        action_output = self.model.model.action_head(fused_embeddings)
        action_dim = self.model.model.action_dim
        action_mean = action_output[:, :action_dim]
        action_log_std = action_output[:, action_dim:]

        return action_mean, action_log_std

    def postprocess(
        self, model_outputs: tuple, return_std: bool = False, deterministic: bool = True, **kwargs
    ) -> Union[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]]:
        """Postprocess model outputs.

        Args:
            model_outputs: Tuple of (action_mean, action_log_std) tensors
            return_std: Whether to include standard deviation in output
            deterministic: If True, return mean actions only. If False, sample from distribution
            **kwargs: Additional postprocessing parameters

        Returns:
            Dictionary or list of dictionaries containing:
                - action: Predicted action (denormalized)
                - action_raw: Predicted action (normalized)
                - action_std: Action std (denormalized, if return_std=True)
                - action_std_raw: Action std (normalized, if return_std=True)
        """
        action_mean, action_log_std = model_outputs

        # Convert to numpy
        action_mean = action_mean.detach().cpu().numpy()
        action_log_std = action_log_std.detach().cpu().numpy()
        action_std = np.exp(action_log_std)

        batch_size = action_mean.shape[0]

        # Prepare outputs
        outputs = []
        for i in range(batch_size):
            result = {
                "action_raw": action_mean[i],
                "action_std_raw": action_std[i] if return_std else None,
            }

            # Denormalize if action processor is available
            if self.action_processor is not None:
                action_mean_tensor = torch.from_numpy(action_mean[i : i + 1])
                action_std_tensor = torch.from_numpy(action_std[i : i + 1])

                action_denorm = self.action_processor.denormalize(action_mean_tensor)
                action_std_denorm = (
                    (
                        action_std_tensor
                        * (self.action_processor.max_val - self.action_processor.min_val)
                        / 2
                    )
                    .detach()
                    .cpu()
                    .numpy()[0]
                )

                result["action"] = action_denorm.detach().cpu().numpy()[0]
                if return_std:
                    result["action_std"] = action_std_denorm
            else:
                result["action"] = action_mean[i]
                if return_std:
                    result["action_std"] = action_std[i]

            # Sample from distribution if not deterministic
            if not deterministic:
                sampled_action = np.random.normal(action_mean[i], action_std[i])
                if self.action_processor is not None:
                    sampled_action_tensor = torch.from_numpy(sampled_action[np.newaxis, :])
                    sampled_action = self.action_processor.denormalize(sampled_action_tensor)
                    sampled_action = sampled_action.detach().cpu().numpy()[0]
                result["action"] = sampled_action

            outputs.append(result)

        return outputs

    def __call__(
        self,
        images: Union[np.ndarray, List[np.ndarray]],
        text: Union[str, List[str]],
        states: Union[np.ndarray, List[np.ndarray]],
        return_std: bool = False,
        deterministic: bool = True,
        **kwargs,
    ) -> Union[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]]:
        """Run VLA policy inference.

        Args:
            images: Single image (H, W, C) or batch of images (B, H, W, C) as numpy arrays
            text: Single instruction string or list of instruction strings
            states: Single state vector or batch of state vectors
            return_std: Whether to return action standard deviation
            deterministic: If True, return mean actions. If False, sample from distribution
            **kwargs: Additional parameters

        Returns:
            Dictionary or list of dictionaries containing predicted actions

        Example:
            >>> image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            >>> state = np.random.randn(14)
            >>> result = pipe(images=image, text="Move forward", states=state, deterministic=True)
            >>> print(result['action'].shape)  # (6,) for 6-DoF action
        """
        inputs = {
            "images": images,
            "text": text,
            "states": states,
        }
        return super().__call__(
            inputs, return_std=return_std, deterministic=deterministic, **kwargs
        )
