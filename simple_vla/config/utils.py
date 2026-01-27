"""Utility functions for building model and training objects from Hydra config."""

from transformers import TrainingArguments

from simple_vla.model.config import SimpleVLAPolicyConfig
from simple_vla.model.simple_policy import SimpleVLAPolicyModel


def build_model_from_cfg(cfg):
    """Build SimpleVLAPolicyModel from Hydra config.

    Args:
        cfg: Hydra config object with model attributes

    Returns:
        SimpleVLAPolicyModel instance
    """
    model_config = SimpleVLAPolicyConfig(
        vision_model_name=cfg.model.vision_model_name,
        text_model_name=cfg.model.text_model_name,
        state_dim=cfg.model.state_dim,
        action_dim=cfg.model.action_dim,
        hidden_dim=cfg.model.hidden_dim,
        freeze_encoder=cfg.model.freeze_encoder,
    )
    model = SimpleVLAPolicyModel(model_config)
    return model


def build_training_arguments_from_cfg(cfg):
    """Build TrainingArguments from Hydra config.

    Args:
        cfg: Hydra config object with training attributes

    Returns:
        TrainingArguments instance
    """
    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        learning_rate=cfg.training.learning_rate,
        warmup_ratio=cfg.training.get("warmup_ratio", 0.1),
        logging_steps=cfg.training.logging_steps,
        save_strategy=cfg.training.save_strategy,
        save_total_limit=cfg.training.save_total_limit,
        remove_unused_columns=cfg.training.remove_unused_columns,
        report_to=cfg.training.get("report_to", None),
    )
    return training_args
