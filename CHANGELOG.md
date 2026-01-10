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

### Changed
- Updated lerobot dependency to 0.4.0
- Added pre-commit and ipdb to dev dependencies

## [0.0.1] - 2025-01-10

### Added
- Initial project structure
- Pre-commit configuration
- Test fixtures matching LeRobot dataset format
