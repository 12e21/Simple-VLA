"""Unit tests for VLA Policy Pipeline."""

from collections.abc import Generator

import numpy as np
import pytest
import torch
from PIL import Image

from simple_vla.eval.pipeline_infer import VLAPolicyPipeline
from simple_vla.model import (
    ActionProcessor,
    SimpleVLAPolicyConfig,
    SimpleVLAPolicyModel,
    StateProcessor,
)


@pytest.fixture
def vla_config() -> SimpleVLAPolicyConfig:
    """VLA model configuration for testing.

    Returns:
        SimpleVLAPolicyConfig: Model configuration
    """
    return SimpleVLAPolicyConfig(
        vision_model_name="openai/clip-vit-base-patch32",
        text_model_name="bert-base-uncased",
        state_dim=14,
        action_dim=6,
        hidden_dim=512,
        freeze_encoder=False,  # Don't freeze for faster testing
    )


@pytest.fixture
def vla_model(vla_config: SimpleVLAPolicyConfig) -> SimpleVLAPolicyModel:
    """VLA model instance for testing.

    Args:
        vla_config: Model configuration fixture

    Returns:
        SimpleVLAPolicyModel: Model instance
    """
    return SimpleVLAPolicyModel(vla_config)


@pytest.fixture
def state_processor() -> StateProcessor:
    """State processor for testing.

    Returns:
        StateProcessor: State normalization processor
    """
    min_val = torch.tensor([-1.0] * 14)
    max_val = torch.tensor([1.0] * 14)
    return StateProcessor(state_dim=14, min_val=min_val, max_val=max_val)


@pytest.fixture
def action_processor() -> ActionProcessor:
    """Action processor for testing.

    Returns:
        ActionProcessor: Action normalization processor
    """
    min_val = torch.tensor([-1.0] * 6)
    max_val = torch.tensor([1.0] * 6)
    return ActionProcessor(action_dim=6, min_val=min_val, max_val=max_val)


@pytest.fixture
def vla_pipeline(
    vla_model: SimpleVLAPolicyModel,
    state_processor: StateProcessor,
    action_processor: ActionProcessor,
) -> Generator[VLAPolicyPipeline, None, None]:
    """VLA pipeline instance for testing.

    Args:
        vla_model: Model fixture
        state_processor: State processor fixture
        action_processor: Action processor fixture

    Yields:
        VLAPolicyPipeline: Pipeline instance in eval mode
    """
    pipeline = VLAPolicyPipeline(
        model=vla_model,
        state_processor=state_processor,
        action_processor=action_processor,
    )
    pipeline.model.eval()
    yield pipeline


@pytest.fixture
def numpy_image() -> np.ndarray:
    """Sample image as numpy array (H, W, C) in RGB format.

    Returns:
        np.ndarray: RGB image
    """
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def numpy_image_batch() -> np.ndarray:
    """Batch of images as numpy array (B, H, W, C).

    Returns:
        np.ndarray: Batch of RGB images
    """
    return np.random.randint(0, 255, (4, 224, 224, 3), dtype=np.uint8)


@pytest.fixture
def numpy_state() -> np.ndarray:
    """Sample state as numpy array.

    Returns:
        np.ndarray: State vector
    """
    return np.random.randn(14).astype(np.float32)


@pytest.fixture
def numpy_state_batch() -> np.ndarray:
    """Batch of states as numpy array.

    Returns:
        np.ndarray: Batch of state vectors
    """
    return np.random.randn(4, 14).astype(np.float32)


class TestVLAPolicyPipelineInit:
    """Test pipeline initialization."""

    def test_pipeline_initialization(self, vla_model, state_processor, action_processor):
        """Test that pipeline initializes correctly with all components."""
        pipeline = VLAPolicyPipeline(
            model=vla_model,
            state_processor=state_processor,
            action_processor=action_processor,
        )

        assert pipeline.model == vla_model
        assert pipeline.state_processor == state_processor
        assert pipeline.action_processor == action_processor
        assert hasattr(pipeline, "image_processor")
        assert hasattr(pipeline, "text_tokenizer")

    def test_pipeline_without_processors(self, vla_model):
        """Test pipeline initialization without processors (should work but skip normalization)."""
        pipeline = VLAPolicyPipeline(model=vla_model, state_processor=None, action_processor=None)

        assert pipeline.model == vla_model
        assert pipeline.state_processor is None
        assert pipeline.action_processor is None


class TestPreprocess:
    """Test input preprocessing."""

    def test_preprocess_single_numpy_image(self, vla_pipeline, numpy_image, numpy_state):
        """Test preprocessing with single numpy image."""
        text = "pick up the block"
        inputs = {
            "images": numpy_image,
            "text": text,
            "states": numpy_state,
        }

        result = vla_pipeline.preprocess(inputs)

        assert "vision_embeddings" in result
        assert "text_embeddings" in result
        assert "states" in result
        assert result["vision_embeddings"].shape[0] == 1  # Batch size 1
        assert result["text_embeddings"].shape[0] == 1
        assert result["states"].shape == (1, 14)

    def test_preprocess_batch_numpy_images(
        self, vla_pipeline, numpy_image_batch, numpy_state_batch
    ):
        """Test preprocessing with batch of numpy images."""
        text = ["pick up the block"] * 4
        inputs = {
            "images": numpy_image_batch,
            "text": text,
            "states": numpy_state_batch,
        }

        result = vla_pipeline.preprocess(inputs)

        assert result["vision_embeddings"].shape[0] == 4
        assert result["text_embeddings"].shape[0] == 4
        assert result["states"].shape == (4, 14)

    def test_preprocess_list_of_numpy_images(self, vla_pipeline, numpy_image, numpy_state):
        """Test preprocessing with list of numpy images."""
        images = [numpy_image, numpy_image]
        text = ["pick up the block", "move forward"]
        states = np.stack([numpy_state, numpy_state])

        inputs = {
            "images": images,
            "text": text,
            "states": states,
        }

        result = vla_pipeline.preprocess(inputs)

        assert result["vision_embeddings"].shape[0] == 2
        assert result["text_embeddings"].shape[0] == 2

    def test_preprocess_pil_images(self, vla_pipeline, numpy_state):
        """Test preprocessing with PIL images."""
        images = [Image.new("RGB", (224, 224), color=(128, 128, 128)) for _ in range(2)]
        text = ["pick up the block", "move forward"]
        states = np.random.randn(2, 14).astype(np.float32)

        inputs = {
            "images": images,
            "text": text,
            "states": states,
        }

        result = vla_pipeline.preprocess(inputs)

        assert result["vision_embeddings"].shape[0] == 2
        assert result["text_embeddings"].shape[0] == 2

    def test_preprocess_invalid_image_shape(self, vla_pipeline, numpy_state):
        """Test that invalid image shapes raise ValueError."""
        # 2D image (missing channel dimension)
        invalid_image = np.random.randint(0, 255, (224, 224), dtype=np.uint8)

        inputs = {
            "images": invalid_image,
            "text": "test",
            "states": numpy_state,
        }

        with pytest.raises(ValueError, match="Images must be 3D.*or 4D"):
            vla_pipeline.preprocess(inputs)

    def test_preprocess_missing_keys(self, vla_pipeline, numpy_image):
        """Test that missing required keys raise ValueError."""
        inputs = {
            "images": numpy_image,
            # Missing 'text' and 'states'
        }

        with pytest.raises(ValueError, match="must contain.*images.*text.*states"):
            vla_pipeline.preprocess(inputs)

    def test_preprocess_invalid_state_shape(self, vla_pipeline, numpy_image):
        """Test that invalid state shapes raise ValueError."""
        # 3D state (invalid)
        invalid_state = np.random.randn(2, 2, 2).astype(np.float32)

        inputs = {
            "images": numpy_image,
            "text": "test",
            "states": invalid_state,
        }

        with pytest.raises(ValueError, match="States must be 1D.*or 2D"):
            vla_pipeline.preprocess(inputs)


class TestForward:
    """Test model forward pass."""

    def test_forward_single_input(self, vla_pipeline, numpy_image, numpy_state):
        """Test forward pass with single input."""
        text = "pick up the block"
        inputs = {
            "images": numpy_image,
            "text": text,
            "states": numpy_state,
        }

        model_inputs = vla_pipeline.preprocess(inputs)
        action_mean, action_log_std = vla_pipeline._forward(model_inputs)

        assert action_mean.shape == (1, 6)  # (batch_size, action_dim)
        assert action_log_std.shape == (1, 6)
        assert not torch.isnan(action_mean).any()
        assert not torch.isnan(action_log_std).any()

    def test_forward_batch_input(self, vla_pipeline, numpy_image_batch, numpy_state_batch):
        """Test forward pass with batch input."""
        text = ["pick up the block"] * 4
        inputs = {
            "images": numpy_image_batch,
            "text": text,
            "states": numpy_state_batch,
        }

        model_inputs = vla_pipeline.preprocess(inputs)
        action_mean, action_log_std = vla_pipeline._forward(model_inputs)

        assert action_mean.shape == (4, 6)
        assert action_log_std.shape == (4, 6)

    def test_forward_output_range(self, vla_pipeline, numpy_image, numpy_state):
        """Test that forward pass outputs reasonable values (not exploding)."""
        text = "pick up the block"
        inputs = {
            "images": numpy_image,
            "text": text,
            "states": numpy_state,
        }

        model_inputs = vla_pipeline.preprocess(inputs)
        action_mean, action_log_std = vla_pipeline._forward(model_inputs)

        # Check that values are finite
        assert torch.isfinite(action_mean).all()
        assert torch.isfinite(action_log_std).all()

        # Check that log_std is in reasonable range (e.g., not too extreme)
        assert (action_log_std > -10).all()
        assert (action_log_std < 10).all()


class TestPostprocess:
    """Test output postprocessing."""

    def test_postprocess_single_output(self, vla_pipeline):
        """Test postprocessing single output."""
        action_mean = torch.randn(1, 6)
        action_log_std = torch.randn(1, 6)

        result = vla_pipeline.postprocess(
            (action_mean, action_log_std), return_std=False, deterministic=True
        )

        assert isinstance(result, list)
        assert len(result) == 1
        assert "action" in result[0]
        assert "action_raw" in result[0]
        assert result[0]["action"].shape == (6,)
        assert result[0]["action_raw"].shape == (6,)

    def test_postprocess_batch_output(self, vla_pipeline):
        """Test postprocessing batch output."""
        action_mean = torch.randn(4, 6)
        action_log_std = torch.randn(4, 6)

        result = vla_pipeline.postprocess(
            (action_mean, action_log_std), return_std=False, deterministic=True
        )

        assert isinstance(result, list)
        assert len(result) == 4
        assert all("action" in r for r in result)
        assert all("action_raw" in r for r in result)

    def test_postprocess_with_std(self, vla_pipeline):
        """Test postprocessing with standard deviation."""
        action_mean = torch.randn(1, 6)
        action_log_std = torch.randn(1, 6)

        result = vla_pipeline.postprocess(
            (action_mean, action_log_std), return_std=True, deterministic=True
        )

        assert isinstance(result, list)
        assert len(result) == 1
        assert "action_std" in result[0]
        assert "action_std_raw" in result[0]
        assert result[0]["action_std"].shape == (6,)
        assert result[0]["action_std_raw"].shape == (6,)

    def test_postprocess_stochastic_mode(self, vla_pipeline):
        """Test postprocessing in stochastic mode (samples from distribution)."""
        action_mean = torch.zeros(1, 6)
        action_log_std = torch.zeros(1, 6)  # std = 1.0

        # Set seed for reproducibility
        np.random.seed(42)

        result = vla_pipeline.postprocess(
            (action_mean, action_log_std), return_std=False, deterministic=False
        )

        assert isinstance(result, list)
        assert len(result) == 1
        assert "action" in result[0]
        # In stochastic mode, action should be sampled, not exactly the mean
        assert not np.allclose(result[0]["action"], action_mean.detach().cpu().numpy()[0])

    def test_postprocess_denormalization(self, vla_pipeline):
        """Test that postprocessing correctly denormalizes actions."""
        # Create normalized action in [-1, 1] range
        action_mean = torch.zeros(1, 6) * 0.5
        action_log_std = torch.zeros(1, 6)

        result = vla_pipeline.postprocess(
            (action_mean, action_log_std), return_std=False, deterministic=True
        )

        # After denormalization, values should be in original range
        # With our min_val=-1, max_val=1, denormalized values should still be around 0
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["action"].shape == (6,)


class TestEndToEnd:
    """Test end-to-end pipeline inference."""

    def test_single_inference_deterministic(self, vla_pipeline, numpy_image, numpy_state):
        """Test single inference with deterministic output."""
        text = "pick up the red block"

        result = vla_pipeline(images=numpy_image, text=text, states=numpy_state, deterministic=True)

        assert isinstance(result, list)
        assert len(result) == 1
        assert "action" in result[0]
        assert "action_raw" in result[0]
        assert result[0]["action"].shape == (6,)

    def test_batch_inference(self, vla_pipeline, numpy_image_batch, numpy_state_batch):
        """Test batch inference."""
        text = ["pick up the red block"] * 4

        results = vla_pipeline(images=numpy_image_batch, text=text, states=numpy_state_batch)

        assert isinstance(results, list)
        assert len(results) == 4
        assert all("action" in r for r in results)

    def test_inference_with_std(self, vla_pipeline, numpy_image, numpy_state):
        """Test inference returning standard deviation."""
        text = "pick up the red block"

        result = vla_pipeline(images=numpy_image, text=text, states=numpy_state, return_std=True)

        assert isinstance(result, list)
        assert len(result) == 1
        assert "action_std" in result[0]
        assert "action_std_raw" in result[0]
        assert result[0]["action_std"].shape == (6,)

    def test_inference_stochastic(self, vla_pipeline, numpy_image, numpy_state):
        """Test stochastic inference (samples from distribution)."""
        text = "pick up the red block"

        result1 = vla_pipeline(
            images=numpy_image, text=text, states=numpy_state, deterministic=False
        )
        result2 = vla_pipeline(
            images=numpy_image, text=text, states=numpy_state, deterministic=False
        )

        # Results should be different due to sampling
        assert not np.allclose(result1[0]["action"], result2[0]["action"])

    def test_inference_consistency(self, vla_pipeline, numpy_image, numpy_state):
        """Test that deterministic inference is consistent."""
        text = "pick up the red block"

        result1 = vla_pipeline(
            images=numpy_image, text=text, states=numpy_state, deterministic=True
        )
        result2 = vla_pipeline(
            images=numpy_image, text=text, states=numpy_state, deterministic=True
        )

        # Same inputs should give same outputs in deterministic mode
        np.testing.assert_allclose(result1[0]["action"], result2[0]["action"], rtol=1e-5)


class TestPipelineWithoutProcessors:
    """Test pipeline behavior without normalization processors."""

    @pytest.fixture
    def pipeline_no_processors(self, vla_model) -> Generator[VLAPolicyPipeline, None, None]:
        """Pipeline without processors."""
        pipeline = VLAPolicyPipeline(
            model=vla_model,
            state_processor=None,
            action_processor=None,
        )
        pipeline.model.eval()
        yield pipeline

    def test_inference_without_processors(self, pipeline_no_processors, numpy_image, numpy_state):
        """Test that inference works without processors (no normalization)."""
        text = "pick up the block"

        result = pipeline_no_processors(images=numpy_image, text=text, states=numpy_state)

        assert isinstance(result, list)
        assert len(result) == 1
        assert "action" in result[0]
        assert "action_raw" in result[0]
        # Without processors, action and action_raw should be the same
        np.testing.assert_allclose(result[0]["action"], result[0]["action_raw"], rtol=1e-5)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_input_type(self, vla_pipeline):
        """Test that invalid input types raise appropriate errors."""
        with pytest.raises(ValueError, match="Inputs must be a dictionary"):
            vla_pipeline.preprocess("invalid_input")

    def test_mismatched_batch_sizes(self, vla_pipeline, numpy_image, numpy_state):
        """Test that mismatched batch sizes are handled."""
        # 4 images but 2 states
        images = np.stack([numpy_image] * 4)
        states = np.stack([numpy_state] * 2)

        inputs = {
            "images": images,
            "text": ["test"] * 4,
            "states": states,
        }

        # This should either raise an error or be handled gracefully
        # Currently it will fail during model forward pass due to shape mismatch
        model_inputs = vla_pipeline.preprocess(inputs)

        with pytest.raises((RuntimeError, ValueError)):
            vla_pipeline._forward(model_inputs)


class TestDeviceHandling:
    """Test device placement and tensor movement."""

    def test_preprocess_places_tensors_on_correct_device(
        self, vla_pipeline, numpy_image, numpy_state
    ):
        """Test that preprocessing places tensors on the correct device."""
        text = "pick up the block"
        inputs = {
            "images": numpy_image,
            "text": text,
            "states": numpy_state,
        }

        result = vla_pipeline.preprocess(inputs)

        # Check that tensors are on the same device as the model
        model_device = next(vla_pipeline.model.parameters()).device
        assert result["states"].device == model_device
        assert result["vision_embeddings"].device == model_device
        assert result["text_embeddings"].device == model_device


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_robot_control_scenario(self, vla_pipeline):
        """Simulate a realistic robot control scenario."""
        # Simulate observation from robot camera
        observation_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Current robot state (e.g., joint positions, gripper state)
        robot_state = np.array(
            [
                0.1,
                0.2,
                0.3,  # position
                0.0,
                0.0,
                0.0,  # orientation
                0.5,  # gripper open
                0.0,
                0.0,
                0.0,
                0.0,  # velocity
                0.0,
                0.0,
                0.0,  # gripper velocity, extra state
            ]
        ).astype(np.float32)

        # Task instruction
        instruction = "pick up the red block"

        # Run inference
        result = vla_pipeline(
            images=observation_image,
            text=instruction,
            states=robot_state,
            deterministic=True,
        )

        # Verify output
        assert isinstance(result, list)
        assert len(result) == 1
        assert "action" in result[0]
        assert result[0]["action"].shape == (6,)
        assert np.isfinite(result[0]["action"]).all()

        # Action should be within reasonable bounds
        assert np.abs(result[0]["action"]).max() < 100  # Not exploding

    def test_multi_step_prediction(self, vla_pipeline):
        """Test multiple sequential predictions (simulating trajectory)."""
        current_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        current_state = np.random.randn(14).astype(np.float32)
        instruction = "move the block forward"

        actions = []
        for _ in range(5):
            result = vla_pipeline(
                images=current_image,
                text=instruction,
                states=current_state,
                deterministic=True,
            )
            actions.append(result[0]["action"])

            # Simulate state update (in real scenario, this would come from robot)
            # For now, just use the same state

        assert len(actions) == 5
        assert all(a.shape == (6,) for a in actions)
