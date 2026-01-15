from transformers import PretrainedConfig


class SimpleVLAPolicyConfig(PretrainedConfig):
    model_type = "simple_vla"

    def __init__(
        self,
        vision_model_name: str = "openai/clip-vit-base-patch32",
        text_model_name: str = "bert-base-uncased",
        state_dim: int = 14,
        action_dim: int = 6,
        hidden_dim: int = 512,
        freeze_encoder: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vision_model_name = vision_model_name
        self.text_model_name = text_model_name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.freeze_encoder = freeze_encoder
