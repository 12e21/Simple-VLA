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
        min_val = torch.tensor([0.0, 0.0, 0.0])
        max_val = torch.tensor([1.0, 1.0, 1.0])
        normalized = normalize(data, min_val, max_val)

        expected = torch.tensor([-1.0, 0.0, 1.0])
        torch.testing.assert_close(normalized, expected)

    def test_normalize_with_per_dim_ranges(self):
        """Should handle per-dimension min/max ranges."""
        data = torch.tensor([0.0, 5.0, 10.0])
        min_val = torch.tensor([0.0, 0.0, 0.0])
        max_val = torch.tensor([1.0, 10.0, 20.0])
        normalized = normalize(data, min_val, max_val)

        expected = torch.tensor([-1.0, 0.0, 0.0])
        torch.testing.assert_close(normalized, expected)

    def test_normalize_batch(self):
        """Should handle batched data."""
        data = torch.tensor([[0.0, 1.0], [0.5, 0.5]])
        min_val = torch.tensor([0.0, 0.0])
        max_val = torch.tensor([1.0, 1.0])
        normalized = normalize(data, min_val, max_val)

        expected = torch.tensor([[-1.0, 1.0], [0.0, 0.0]])
        torch.testing.assert_close(normalized, expected)


class TestDenormalize:
    """Tests for denormalize function."""

    def test_denormalize_from_range_minus1_to_1(self):
        """Should denormalize data from [-1, 1] to original range."""
        data = torch.tensor([-1.0, 0.0, 1.0])
        min_val = torch.tensor([0.0, 0.0, 0.0])
        max_val = torch.tensor([1.0, 1.0, 1.0])
        denormalized = denormalize(data, min_val, max_val)

        expected = torch.tensor([0.0, 0.5, 1.0])
        torch.testing.assert_close(denormalized, expected)

    def test_denormalize_with_per_dim_ranges(self):
        """Should handle per-dimension min/max ranges."""
        data = torch.tensor([-1.0, 0.0, 1.0])
        min_val = torch.tensor([-10.0, 0.0, -5.0])
        max_val = torch.tensor([10.0, 10.0, 5.0])
        denormalized = denormalize(data, min_val, max_val)

        expected = torch.tensor([-10.0, 5.0, 5.0])
        torch.testing.assert_close(denormalized, expected)

    def test_normalize_denormalize_roundtrip(self):
        """Should return original values after normalize+denormalize."""
        original = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        min_val = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
        max_val = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        normalized = normalize(original, min_val, max_val)
        denormalized = denormalize(normalized, min_val, max_val)

        torch.testing.assert_close(denormalized, original)

    def test_normalize_denormalize_batch(self):
        """Should handle batched data in roundtrip."""
        original = torch.tensor([[0.0, 1.0], [0.25, 0.75]])
        min_val = torch.tensor([0.0, 0.0])
        max_val = torch.tensor([1.0, 1.0])
        normalized = normalize(original, min_val, max_val)
        denormalized = denormalize(normalized, min_val, max_val)

        torch.testing.assert_close(denormalized, original)


class TestStateProcessor:
    """Tests for StateProcessor."""

    def test_init(self):
        """Should initialize with tensor parameters."""
        state_dim = 2
        min_val = torch.tensor([0.0, 0.0])
        max_val = torch.tensor([10.0, 10.0])
        processor = StateProcessor(state_dim, min_val, max_val)

        assert processor.state_dim == state_dim
        torch.testing.assert_close(processor.min_val, min_val)
        torch.testing.assert_close(processor.max_val, max_val)

    def test_normalize_single_state(self):
        """Should normalize single state with per-dim ranges."""
        state_dim = 3
        min_val = torch.tensor([0.0, 0.0, 0.0])
        max_val = torch.tensor([1.0, 10.0, 100.0])
        processor = StateProcessor(state_dim, min_val, max_val)
        state = torch.tensor([0.5, 5.0, 50.0])

        normalized = processor.normalize(state)

        expected = torch.tensor([0.0, 0.0, 0.0])
        torch.testing.assert_close(normalized, expected)

    def test_normalize_batch_states(self):
        """Should normalize batch of states."""
        state_dim = 2
        min_val = torch.tensor([0.0, 0.0])
        max_val = torch.tensor([10.0, 10.0])
        processor = StateProcessor(state_dim, min_val, max_val)
        batch = torch.tensor([[0.0, 10.0], [5.0, 2.5]])

        normalized = processor.normalize(batch)

        expected = torch.tensor([[-1.0, 1.0], [0.0, -0.5]])
        torch.testing.assert_close(normalized, expected)

    def test_denormalize_single_state(self):
        """Should denormalize single state."""
        state_dim = 2
        min_val = torch.tensor([0.0, 0.0])
        max_val = torch.tensor([10.0, 10.0])
        processor = StateProcessor(state_dim, min_val, max_val)
        normalized_state = torch.tensor([-1.0, 1.0])

        denormalized = processor.denormalize(normalized_state)

        expected = torch.tensor([0.0, 10.0])
        torch.testing.assert_close(denormalized, expected)

    def test_denormalize_batch_states(self):
        """Should denormalize batch of states."""
        state_dim = 3
        min_val = torch.tensor([-5.0, -5.0, -5.0])
        max_val = torch.tensor([5.0, 5.0, 5.0])
        processor = StateProcessor(state_dim, min_val, max_val)
        normalized_batch = torch.tensor([[-1.0, 0.0, 1.0], [0.0, -0.5, 0.5]])

        denormalized = processor.denormalize(normalized_batch)

        expected = torch.tensor([[-5.0, 0.0, 5.0], [0.0, -2.5, 2.5]])
        torch.testing.assert_close(denormalized, expected)

    def test_normalize_denormalize_single_state_roundtrip(self):
        """Should return original single state after normalize+denormalize."""
        state_dim = 4
        min_val = torch.tensor([-1.0, -1.0, -1.0, -1.0])
        max_val = torch.tensor([1.0, 1.0, 1.0, 1.0])
        processor = StateProcessor(state_dim, min_val, max_val)
        original = torch.tensor([0.0, 0.5, -0.5, 1.0])

        normalized = processor.normalize(original)
        denormalized = processor.denormalize(normalized)

        torch.testing.assert_close(denormalized, original)

    def test_normalize_denormalize_batch_states_roundtrip(self):
        """Should return original batch states after normalize+denormalize."""
        state_dim = 2
        min_val = torch.tensor([0.0, 0.0])
        max_val = torch.tensor([10.0, 10.0])
        processor = StateProcessor(state_dim, min_val, max_val)
        original_batch = torch.tensor([[0.0, 10.0], [5.0, 2.5], [7.5, 3.0]])

        normalized = processor.normalize(original_batch)
        denormalized = processor.denormalize(normalized)

        torch.testing.assert_close(denormalized, original_batch)


class TestActionProcessor:
    """Tests for ActionProcessor."""

    def test_init(self):
        """Should initialize with tensor parameters."""
        action_dim = 2
        min_val = torch.tensor([-1.0, -1.0])
        max_val = torch.tensor([1.0, 1.0])
        processor = ActionProcessor(action_dim, min_val, max_val)

        assert processor.action_dim == action_dim
        torch.testing.assert_close(processor.min_val, min_val)
        torch.testing.assert_close(processor.max_val, max_val)

    def test_normalize_single_action(self):
        """Should normalize single action with per-dim ranges."""
        action_dim = 2
        min_val = torch.tensor([-3.0, -3.0])
        max_val = torch.tensor([3.0, 3.0])
        processor = ActionProcessor(action_dim, min_val, max_val)
        action = torch.tensor([-3.0, 3.0])

        normalized = processor.normalize(action)

        expected = torch.tensor([-1.0, 1.0])
        torch.testing.assert_close(normalized, expected)

    def test_normalize_batch_actions(self):
        """Should normalize batch of actions."""
        action_dim = 3
        min_val = torch.tensor([-2.0, -2.0, -2.0])
        max_val = torch.tensor([2.0, 2.0, 2.0])
        processor = ActionProcessor(action_dim, min_val, max_val)
        batch = torch.tensor([[0.0, 1.0, 2.0], [-1.0, 0.0, -2.0]])

        normalized = processor.normalize(batch)

        expected = torch.tensor([[0.0, 0.5, 1.0], [-0.5, 0.0, -1.0]])
        torch.testing.assert_close(normalized, expected)

    def test_denormalize_single_action(self):
        """Should denormalize single action."""
        action_dim = 2
        min_val = torch.tensor([-3.0, -3.0])
        max_val = torch.tensor([3.0, 3.0])
        processor = ActionProcessor(action_dim, min_val, max_val)
        normalized_action = torch.tensor([-1.0, 1.0])

        denormalized = processor.denormalize(normalized_action)

        expected = torch.tensor([-3.0, 3.0])
        torch.testing.assert_close(denormalized, expected)

    def test_denormalize_batch_actions(self):
        """Should denormalize batch of actions."""
        action_dim = 4
        min_val = torch.tensor([-1.0, -1.0, -1.0, -1.0])
        max_val = torch.tensor([1.0, 1.0, 1.0, 1.0])
        processor = ActionProcessor(action_dim, min_val, max_val)
        normalized_batch = torch.tensor([[-1.0, 0.0, 1.0, 0.5], [0.0, -0.5, 0.5, -1.0]])

        denormalized = processor.denormalize(normalized_batch)

        expected = torch.tensor([[-1.0, 0.0, 1.0, 0.5], [0.0, -0.5, 0.5, -1.0]])
        torch.testing.assert_close(denormalized, expected)

    def test_normalize_denormalize_single_action_roundtrip(self):
        """Should return original single action after normalize+denormalize."""
        action_dim = 3
        min_val = torch.tensor([-1.0, -1.0, -1.0])
        max_val = torch.tensor([1.0, 1.0, 1.0])
        processor = ActionProcessor(action_dim, min_val, max_val)
        original = torch.tensor([0.0, 0.5, -0.5])

        normalized = processor.normalize(original)
        denormalized = processor.denormalize(normalized)

        torch.testing.assert_close(denormalized, original)

    def test_normalize_denormalize_batch_actions_roundtrip(self):
        """Should return original batch actions after normalize+denormalize."""
        action_dim = 2
        min_val = torch.tensor([-3.0, -3.0])
        max_val = torch.tensor([3.0, 3.0])
        processor = ActionProcessor(action_dim, min_val, max_val)
        original_batch = torch.tensor([[0.0, 3.0], [-1.5, 1.5], [-3.0, 0.0]])

        normalized = processor.normalize(original_batch)
        denormalized = processor.denormalize(normalized)

        torch.testing.assert_close(denormalized, original_batch)
