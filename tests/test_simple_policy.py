import torch

from simple_vla.model.simple_policy import SimpleVLAPolicy


class TestSimpleVLAPolicyEncodeImage:
    """Tests for encode_image method."""

    def test_encode_image_returns_tensor(self, dummy_image_batch):
        """Should return a tensor output from vision encoder."""
        model = SimpleVLAPolicy()
        output = model.encode_image(dummy_image_batch)

        assert isinstance(output, torch.Tensor), "encode_image should return a torch.Tensor"

    def test_encode_image_output_shape(self, dummy_image_batch):
        """Should return tensor with correct shape [batch_size, vision_dim]."""
        model = SimpleVLAPolicy()
        batch_size = len(dummy_image_batch)
        vision_dim = model.vision_encoder.config.hidden_size

        output = model.encode_image(dummy_image_batch)

        assert output.shape == (batch_size, vision_dim), (
            f"Expected shape ({batch_size}, {vision_dim}), got {output.shape}"
        )

    def test_encode_image_single_image(self, single_image):
        """Should handle single PIL image input."""
        model = SimpleVLAPolicy()
        vision_dim = model.vision_encoder.config.hidden_size

        output = model.encode_image(single_image)

        assert output.shape == (1, vision_dim), (
            f"Expected shape (1, {vision_dim}), got {output.shape}"
        )


class TestSimpleVLAPolicyEncodeText:
    """Tests for encode_text method."""

    def test_encode_text_returns_tensor(self, dummy_text_batch):
        """Should return a tensor output from text encoder."""
        model = SimpleVLAPolicy()
        output = model.encode_text(dummy_text_batch)

        assert isinstance(output, torch.Tensor), "encode_text should return a torch.Tensor"

    def test_encode_text_output_shape(self, dummy_text_batch):
        """Should return tensor with correct shape [batch_size, text_dim]."""
        model = SimpleVLAPolicy()
        batch_size = len(dummy_text_batch)
        text_dim = model.text_encoder.config.hidden_size

        output = model.encode_text(dummy_text_batch)

        assert output.shape == (batch_size, text_dim), (
            f"Expected shape ({batch_size}, {text_dim}), got {output.shape}"
        )

    def test_encode_text_single_text(self, single_text):
        """Should handle single text input as list of one string."""
        model = SimpleVLAPolicy()
        text_dim = model.text_encoder.config.hidden_size

        output = model.encode_text(single_text)

        assert output.shape == (1, text_dim), f"Expected shape (1, {text_dim}), got {output.shape}"


class TestSimpleVLAPolicyForward:
    """Tests for forward method."""

    def test_forward_returns_action_tuple(
        self, dummy_image_batch, dummy_text_batch, dummy_state_batch
    ):
        """Should return tuple of (action_mean, action_log_std)."""
        model = SimpleVLAPolicy()
        output = model.forward(dummy_image_batch, dummy_text_batch, dummy_state_batch)

        assert isinstance(output, tuple), "forward should return a tuple"
        assert len(output) == 2, "forward should return (action_mean, action_log_std)"
        assert isinstance(output[0], torch.Tensor), "action_mean should be a tensor"
        assert isinstance(output[1], torch.Tensor), "action_log_std should be a tensor"

    def test_forward_action_shape(self, dummy_image_batch, dummy_text_batch, dummy_state_batch):
        """Should return actions with shape [batch_size, action_dim]."""
        model = SimpleVLAPolicy(action_dim=6)
        batch_size = len(dummy_image_batch)

        action_mean, action_log_std = model.forward(
            dummy_image_batch, dummy_text_batch, dummy_state_batch
        )

        assert action_mean.shape == (batch_size, 6), (
            f"Expected action_mean shape ({batch_size}, 6), got {action_mean.shape}"
        )
        assert action_log_std.shape == (batch_size, 6), (
            f"Expected action_log_std shape ({batch_size}, 6), got {action_log_std.shape}"
        )

    def test_forward_fuses_vision_and_text(
        self, dummy_image_batch, dummy_text_batch, dummy_state_batch
    ):
        """Should fuse vision and text encodings before action prediction."""
        model = SimpleVLAPolicy(action_dim=6)

        # Forward should use both encodings
        action_mean, _ = model.forward(dummy_image_batch, dummy_text_batch, dummy_state_batch)

        # Action should depend on both modalities (not all same values)
        assert not torch.allclose(action_mean, action_mean[0:1].expand_as(action_mean)), (
            "Actions should vary based on different vision/text inputs"
        )

    def test_forward_different_batch_sizes(
        self, dummy_image_batch, dummy_text_batch, dummy_state_batch
    ):
        """Should handle different batch sizes correctly."""
        model = SimpleVLAPolicy(action_dim=6)

        # Test with batch size of 4
        images_4 = dummy_image_batch[:4]
        texts_4 = dummy_text_batch[:4]
        states_4 = dummy_state_batch[:4]
        action_mean, action_log_std = model.forward(images_4, texts_4, states_4)

        assert action_mean.shape[0] == 4, "Batch size should be 4"
        assert action_log_std.shape[0] == 4, "Batch size should be 4"
