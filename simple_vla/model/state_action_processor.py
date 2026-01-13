import torch


def normalize(data: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor) -> torch.Tensor:
    """Normalize data to [-1, 1] range using min-max scaling.

    Args:
        data: Input tensor to normalize, shape [..., dim]
        min_val: Minimum values for each dimension, shape [dim]
        max_val: Maximum values for each dimension, shape [dim]

    Returns:
        Normalized tensor in [-1, 1] range, shape [..., dim]
    """
    return 2 * (data - min_val) / (max_val - min_val) - 1


def denormalize(data: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor) -> torch.Tensor:
    """Denormalize data from [-1, 1] range back to original range.

    Args:
        data: Normalized tensor in [-1, 1] range, shape [..., dim]
        min_val: Minimum values for each dimension, shape [dim]
        max_val: Maximum values for each dimension, shape [dim]

    Returns:
        Denormalized tensor in original range, shape [..., dim]
    """
    return (data + 1) / 2 * (max_val - min_val) + min_val


class StateProcessor:
    """Process and normalize state vectors."""

    def __init__(self, state_dim: int, min_val: torch.Tensor, max_val: torch.Tensor):
        """Initialize state processor.

        Args:
            state_dim: Dimension of state vector
            min_val: Minimum values for each dimension, shape [state_dim]
            max_val: Maximum values for each dimension, shape [state_dim]
        """

        assert isinstance(min_val, torch.Tensor) and min_val.shape == (state_dim,)
        assert isinstance(max_val, torch.Tensor) and max_val.shape == (state_dim,)
        self.state_dim = state_dim
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, state: torch.Tensor) -> torch.Tensor:
        """Normalize state to [-1, 1] range.

        Args:
            state: State tensor of shape [..., state_dim]

        Returns:
            Normalized state tensor
        """
        return normalize(state, self.min_val, self.max_val)

    def denormalize(self, state: torch.Tensor) -> torch.Tensor:
        """Denormalize state from [-1, 1] range.

        Args:
            state: Normalized state tensor of shape [..., state_dim]

        Returns:
            Denormalized state tensor
        """
        return denormalize(state, self.min_val, self.max_val)


class ActionProcessor:
    """Process and normalize action vectors."""

    def __init__(self, action_dim: int, min_val: torch.Tensor, max_val: torch.Tensor):
        """Initialize action processor.

        Args:
            action_dim: Dimension of action vector
            min_val: Minimum values for each dimension, shape [action_dim]
            max_val: Maximum values for each dimension, shape [action_dim]
        """
        assert isinstance(min_val, torch.Tensor) and min_val.shape == (action_dim,)
        assert isinstance(max_val, torch.Tensor) and max_val.shape == (action_dim,)
        self.action_dim = action_dim
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, action: torch.Tensor) -> torch.Tensor:
        """Normalize action to [-1, 1] range.

        Args:
            action: Action tensor of shape [..., action_dim]

        Returns:
            Normalized action tensor
        """
        return normalize(action, self.min_val, self.max_val)

    def denormalize(self, action: torch.Tensor) -> torch.Tensor:
        """Denormalize action from [-1, 1] range.

        Args:
            action: Normalized action tensor of shape [..., action_dim]

        Returns:
            Denormalized action tensor
        """
        return denormalize(action, self.min_val, self.max_val)
