import torch


def gaussian_nll_loss(mean, log_std, target):
    # target: (B, action_dim)
    var = torch.exp(2 * log_std)
    return 0.5 * (((target - mean) ** 2) / (var + 1e-8) + 2 * log_std).mean()
