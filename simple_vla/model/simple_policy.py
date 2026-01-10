import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor, CLIPVisionModel


class SimpleVLAPolicy(nn.Module):
    def __init__(
        self,
        vision_model_name: str = "openai/clip-vit-base-patch32",
        text_model_name: str = "bert-base-uncased",
        state_dim: int = 14,
        action_dim: int = 7,
        hidden_dim: int = 512,
        freeze_encoder: bool = True,
    ):
        super().__init__()

        self.image_processor = CLIPImageProcessor.from_pretrained(vision_model_name)
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)

        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)

        vision_dim = self.vision_encoder.config.hidden_size
        text_dim = self.text_encoder.config.hidden_size

        self.state_encoder = nn.Sequential(
            nn.Linear(vision_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        self.fuse_layers = nn.Sequential(
            nn.Linear(vision_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # action gaussian head: output mean and log_std
        self.action_head = nn.Linear(hidden_dim, action_dim * 2)

        if freeze_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            for param in self.text_encoder.parameters():
                param.requires_grad = False

    def encode_image(self, images):
        pass

    def encode_text(self, texts):
        pass

    def forward(self, images, texts):
        pass
