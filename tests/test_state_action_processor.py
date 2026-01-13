import torch

from simple_vla.model.state_action_processor import (
    ActionProcessor,
    StateProcessor,
    denormalize,
    normalize,
)


class TestNormalize:
    """Tests for normalize function."""

    def test_normalize_to_range_minus1_to_1(self):
        """Should normalize data to [-1, 1] range."""
        data = torch.tensor([0.0, 0.5, 1.0])
        normalized = normalize(data, 0.0, 1.0)

        expected = torch.tensor([-1.0, 0.0, 1.0])
        torch.testing.assert_close(normalized, expected)

    def test_normalize_with_custom_range(self):
        """Should handle custom min/max ranges."""
        data = torch.tensor([-5.0, 0.0, 5.0])
        normalized = normalize(data, -5.0, 5.0)

        expected = torch.tensor([-1.0, 0.0, 1.0])
        torch.testing.assert_close(normalized, expected)

    def test_normalize_batch(self):
        """Should handle batched data."""
        data = torch.tensor([[0.0, 1.0], [0.5, 0.5]])
        normalized = normalize(data, 0.0, 1.0)

        expected = torch.tensor([[-1.0, 1.0], [0.0, 0.0]])
        torch.testing.assert_close(normalized, expected)


class TestDenormalize:
    """Tests for denormalize function."""

    def test_denormalize_from_range_minus1_to_1(self):
        """Should denormalize data from [-1, 1] to original range."""
        data = torch.tensor([-1.0, 0.0, 1.0])
        denormalized = denormalize(data, 0.0, 1.0)

        expected = torch.tensor([0.0, 0.5, 1.0])
        torch.testing.assert_close(denormalized, expected)

    def test_denormalize_with_custom_range(self):
        """Should handle custom min/max ranges."""
        data = torch.tensor([-1.0, 0.0, 1.0])
        denormalized = denormalize(data, -5.0, 5.0)

        expected = torch.tensor([-5.0, 0.0, 5.0])
        torch.testing.assert_close(denormalized, expected)

    def test_normalize_denormalize_roundtrip(self):
        """Should return original values after normalize+denormalize."""
        original = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        normalized = normalize(original, 0.0, 1.0)
        denormalized = denormalize(normalized, 0.0, 1.0)

        torch.testing.assert_close(denormalized, original)

    def test_normalize_denormalize_batch(self):
        """Should handle batched data in roundtrip."""
        original = torch.tensor([[0.0, 1.0], [0.25, 0.75]])
        normalized = normalize(original, 0.0, 1.0)
        denormalized = denormalize(normalized, 0.0, 1.0)

        torch.testing.assert_close(denormalized, original)


class TestStateProcessor:
    """Tests for StateProcessor."""

    def test_init_default_params(self):
        """Should initialize with default parameters."""
        processor = StateProcessor(state_dim=14)

        assert processor.state_dim == 14
        assert processor.min_val == -1.0
        assert processor.max_val == 1.0

    def test_init_custom_params(self):
        """Should initialize with custom parameters."""
        processor = StateProcessor(state_dim=10, min_val=-5.0, max_val=5.0)

        assert processor.state_dim == 10
        assert processor.min_val == -5.0
        assert processor.max_val == 5.0

    def test_normalize_single_state(self):
        """Should normalize single state to [-1, 1] range."""
        processor = StateProcessor(state_dim=2, min_val=0.0, max_val=10.0)
        state = torch.tensor([0.0, 10.0])

        normalized = processor.normalize(state)

        expected = torch.tensor([-1.0, 1.0])
        torch.testing.assert_close(normalized, expected)

    def test_normalize_batch_states(self):
        """Should normalize batch of states."""
        processor = StateProcessor(state_dim=3, min_val=-1.0, max_val=1.0)
        batch = torch.tensor([[0.0, 0.5, 1.0], [-0.5, 0.0, 0.5]])

        normalized = processor.normalize(batch)

        expected = torch.tensor([[0.0, 0.5, 1.0], [-0.5, 0.0, 0.5]])
        torch.testing.assert_close(normalized, expected)

    def test_denormalize_single_state(self):
        """Should denormalize single state from [-1, 1] range."""
        processor = StateProcessor(state_dim=2, min_val=0.0, max_val=10.0)
        normalized_state = torch.tensor([-1.0, 1.0])

        denormalized = processor.denormalize(normalized_state)

        expected = torch.tensor([0.0, 10.0])
        torch.testing.assert_close(denormalized, expected)

    def test_denormalize_batch_states(self):
        """Should denormalize batch of states."""
        processor = StateProcessor(state_dim=3, min_val=-5.0, max_val=5.0)
        normalized_batch = torch.tensor([[-1.0, 0.0, 1.0], [0.0, -0.5, 0.5]])

        denormalized = processor.denormalize(normalized_batch)

        expected = torch.tensor([[-5.0, 0.0, 5.0], [0.0, -2.5, 2.5]])
        torch.testing.assert_close(denormalized, expected)

    def test_normalize_denormalize_single_state_roundtrip(self):
        """Should return original single state after normalize+denormalize."""
        processor = StateProcessor(state_dim=4, min_val=-1.0, max_val=1.0)
        original = torch.tensor([0.0, 0.5, -0.5, 1.0])

        normalized = processor.normalize(original)
        denormalized = processor.denormalize(normalized)

        torch.testing.assert_close(denormalized, original)

    def test_normalize_denormalize_batch_states_roundtrip(self):
        """Should return original batch states after normalize+denormalize."""
        processor = StateProcessor(state_dim=2, min_val=0.0, max_val=10.0)
        original_batch = torch.tensor([[0.0, 10.0], [5.0, 2.5], [7.5, 3.0]])

        normalized = processor.normalize(original_batch)
        denormalized = processor.denormalize(normalized)

        torch.testing.assert_close(denormalized, original_batch)


class TestActionProcessor:
    """Tests for ActionProcessor."""

    def test_init_default_params(self):
        """Should initialize with default parameters."""
        processor = ActionProcessor(action_dim=6)

        assert processor.action_dim == 6
        assert processor.min_val == -1.0
        assert processor.max_val == 1.0

    def test_init_custom_params(self):
        """Should initialize with custom parameters."""
        processor = ActionProcessor(action_dim=7, min_val=-2.0, max_val=2.0)

        assert processor.action_dim == 7
        assert processor.min_val == -2.0
        assert processor.max_val == 2.0

    def test_normalize_single_action(self):
        """Should normalize single action to [-1, 1] range."""
        processor = ActionProcessor(action_dim=2, min_val=-3.0, max_val=3.0)
        action = torch.tensor([-3.0, 3.0])

        normalized = processor.normalize(action)

        expected = torch.tensor([-1.0, 1.0])
        torch.testing.assert_close(normalized, expected)

    def test_normalize_batch_actions(self):
        """Should normalize batch of actions."""
        processor = ActionProcessor(action_dim=3, min_val=-2.0, max_val=2.0)
        batch = torch.tensor([[0.0, 1.0, 2.0], [-1.0, 0.0, -2.0]])

        normalized = processor.normalize(batch)

        expected = torch.tensor([[0.0, 0.5, 1.0], [-0.5, 0.0, -1.0]])
        torch.testing.assert_close(normalized, expected)

    def test_denormalize_single_action(self):
        """Should denormalize single action from [-1, 1] range."""
        processor = ActionProcessor(action_dim=2, min_val=-3.0, max_val=3.0)
        normalized_action = torch.tensor([-1.0, 1.0])

        denormalized = processor.denormalize(normalized_action)

        expected = torch.tensor([-3.0, 3.0])
        torch.testing.assert_close(denormalized, expected)

    def test_denormalize_batch_actions(self):
        """Should denormalize batch of actions."""
        processor = ActionProcessor(action_dim=4, min_val=-1.0, max_val=1.0)
        normalized_batch = torch.tensor([[-1.0, 0.0, 1.0, 0.5], [0.0, -0.5, 0.5, -1.0]])

        denormalized = processor.denormalize(normalized_batch)

        expected = torch.tensor([[-1.0, 0.0, 1.0, 0.5], [0.0, -0.5, 0.5, -1.0]])
        torch.testing.assert_close(denormalized, expected)

    def test_normalize_denormalize_single_action_roundtrip(self):
        """Should return original single action after normalize+denormalize."""
        processor = ActionProcessor(action_dim=3, min_val=-1.0, max_val=1.0)
        original = torch.tensor([0.0, 0.5, -0.5])

        normalized = processor.normalize(original)
        denormalized = processor.denormalize(normalized)

        torch.testing.assert_close(denormalized, original)

    def test_normalize_denormalize_batch_actions_roundtrip(self):
        """Should return original batch actions after normalize+denormalize."""
        processor = ActionProcessor(action_dim=2, min_val=-3.0, max_val=3.0)
        original_batch = torch.tensor([[0.0, 3.0], [-1.5, 1.5], [-3.0, 0.0]])

        normalized = processor.normalize(original_batch)
        denormalized = processor.denormalize(normalized)

        torch.testing.assert_close(denormalized, original_batch)
