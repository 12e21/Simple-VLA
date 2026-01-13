import torch

from simple_vla.model.loss import gaussian_nll_loss


class TestGaussianNLLLoss:
    """Tests for gaussian_nll_loss function."""

    def test_output_shape(self):
        """Should return scalar loss value."""
        batch_size = 8
        action_dim = 6
        mean = torch.randn(batch_size, action_dim)
        log_std = torch.randn(batch_size, action_dim)
        target = torch.randn(batch_size, action_dim)

        loss = gaussian_nll_loss(mean, log_std, target)

        assert loss.ndim == 0
        assert loss.shape == torch.Size([])

    def test_perfect_prediction(self):
        """Should return low loss when mean matches target."""
        batch_size = 4
        action_dim = 3
        target = torch.randn(batch_size, action_dim)
        mean = target.clone()
        log_std = torch.full_like(mean, -3.0)

        loss = gaussian_nll_loss(mean, log_std, target)

        assert loss.item() < 0

    def test_different_batch_sizes(self):
        """Should handle different batch sizes."""
        action_dim = 6
        target = torch.randn(1, action_dim)

        for batch_size in [1, 4, 8]:
            mean = torch.randn(batch_size, action_dim)
            log_std = torch.randn(batch_size, action_dim)
            loss = gaussian_nll_loss(mean, log_std, target.repeat(batch_size, 1))

            assert loss.ndim == 0

    def test_gradient_flow(self):
        """Should allow gradients to flow through."""
        batch_size = 4
        action_dim = 3
        mean = torch.randn(batch_size, action_dim, requires_grad=True)
        log_std = torch.randn(batch_size, action_dim, requires_grad=True)
        target = torch.randn(batch_size, action_dim)

        loss = gaussian_nll_loss(mean, log_std, target)
        loss.backward()

        assert mean.grad is not None
        assert log_std.grad is not None
