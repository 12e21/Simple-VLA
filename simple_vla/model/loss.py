import torch


def mse_loss(pred, target):
    """Mean squared error loss for deterministic action prediction.

    Args:
        pred: Predicted actions, shape (B, action_dim)
        target: Target actions, shape (B, action_dim)

    Returns:
        Scalar MSE loss
    """
    return torch.nn.functional.mse_loss(pred, target)


def gaussian_nll_loss(mean, log_std, target):
    # target: (B, action_dim)
    var = torch.exp(2 * log_std)
    return 0.5 * (((target - mean) ** 2) / (var + 1e-8) + 2 * log_std).mean()
