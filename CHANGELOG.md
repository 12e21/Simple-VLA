# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- CLAUDE.md with project instructions and development workflow
- CHANGELOG.md for tracking project changes
- SimpleVLAPolicy model skeleton with CLIP vision and BERT text encoders
- TDD tests for encode_image, encode_text, and forward methods
- Test fixtures matching LeRobot dataset format in conftest.py
- .pre-commit-config.yaml for code quality enforcement
- Training script simple_vla/experiment/train.py for VLA model
- gaussian_nll_loss function for training stochastic policy
- mse_loss function for deterministic action prediction and evaluation metrics
- SimpleVLAPolicyConfig for Transformers-style configuration management
- SimpleVLAPolicyModel(PreTrainedModel) for Transformers ecosystem integration
- VLATrainer with custom gaussian_nll_loss compute_loss override and MSE logging
- VLACollator for multi-modal data processing with normalization
- train_trainer.py script for Transformers-style training pipeline

### Changed
- Updated lerobot dependency to 0.4.0
- Added pre-commit and ipdb to dev dependencies
- SimpleVLAPolicy action_dim from 7 to 6
- Image test fixtures from torch.Tensor to PIL.Image
- SimpleVLAPolicy forward method to include state parameter
- normalize/denormalize to support per-dimension min/max values
- StateProcessor and ActionProcessor to accept tensor min_val/max_val
- SimpleVLAPolicy encode methods to handle device placement
- Export StateProcessor and ActionProcessor from model module
- encode_image/encode_text to use device from model parameters instead of hardcoded "cuda"
- VLACollator ensures float32 dtype for model compatibility
- Export SimpleVLAPolicyConfig from model module
- .gitignore to exclude CLAUDE.md files in subdirectories while keeping root CLAUDE.md

### Implemented
- SimpleVLAPolicy.encode_image using CLIP vision encoder
- SimpleVLAPolicy.encode_text using BERT text encoder
- SimpleVLAPolicy.forward method with vision, text, and state fusion
- single_state and dummy_state_batch test fixtures
- Per-dimension normalization in StateProcessor and ActionProcessor
- Training pipeline with LeRobot dataset integration
- Comprehensive tests for state/action processors with per-dim ranges
- Unit tests for gaussian_nll_loss and mse_loss functions

## [0.0.1] - 2025-01-10

### Added
- Initial project structure
- Pre-commit configuration
- Test fixtures matching LeRobot dataset format
