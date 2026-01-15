import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPVisionModel,
    PreTrainedModel,
)

from simple_vla.model.config import SimpleVLAPolicyConfig


class SimpleVLAPolicy(nn.Module):
    def __init__(
        self,
        vision_model_name: str = "openai/clip-vit-base-patch32",
        text_model_name: str = "bert-base-uncased",
        state_dim: int = 14,
        action_dim: int = 6,
        hidden_dim: int = 512,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.image_processor = CLIPImageProcessor.from_pretrained(vision_model_name)
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)

        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)

        self.action_dim = action_dim

        vision_dim = self.vision_encoder.config.hidden_size
        text_dim = self.text_encoder.config.hidden_size

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.fuse_layers = nn.Sequential(
            nn.Linear(vision_dim + text_dim + hidden_dim, hidden_dim),
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
        # Determine device from model parameters
        device = next(self.vision_encoder.parameters()).device
        images = self.image_processor(images, return_tensors="pt").to(device)
        outputs = self.vision_encoder(**images)
        return outputs.pooler_output

    def encode_text(self, texts):
        # Determine device from model parameters
        device = next(self.text_encoder.parameters()).device
        inputs = self.text_tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(
            device
        )
        outputs = self.text_encoder(**inputs)
        return outputs.pooler_output

    def forward(self, images, texts, states):
        state_embeddings = self.state_encoder(states)
        vision_embeddings = self.encode_image(images)
        text_embeddings = self.encode_text(texts)

        fused_embeddings = self.fuse_layers(
            torch.cat((vision_embeddings, text_embeddings, state_embeddings), dim=-1)
        )
        action = self.action_head(fused_embeddings)
        action_mean = action[:, : self.action_dim]
        action_log_std = action[:, self.action_dim :]
        return action_mean, action_log_std


class SimpleVLAPolicyModel(PreTrainedModel):
    config_class = SimpleVLAPolicyConfig

    def __init__(self, config: SimpleVLAPolicyConfig):
        super().__init__(config)
        self.model = SimpleVLAPolicy(
            vision_model_name=config.vision_model_name,
            text_model_name=config.text_model_name,
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            freeze_encoder=config.freeze_encoder,
        )

    def forward(self, images, texts, states):
        return self.model(images, texts, states)
