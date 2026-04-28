import torch
from torch import nn

from .encoder import PocketEncoder
from .predictor import LatentPredictor


class JEPAModel(nn.Module):
    def __init__(self, in_channels=1, latent_dim=128, predictor_hidden_dim=256):
        super().__init__()
        # context encoder, target encoder, and predictor
        self.context_encoder = PocketEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
        )
        self.target_encoder = PocketEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
        )
        self.predictor = LatentPredictor(
            latent_dim=latent_dim,
            hidden_dim=predictor_hidden_dim,
        )

        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    # update target encoder with momentumed average of context encoder parameters
    @torch.no_grad()
    def update_target_network(self, momentum=0.99):
        for context_param, target_param in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            target_param.data.mul_(momentum).add_(
                context_param.data,
                alpha=1.0 - momentum,
            )

    def forward(self, x_context, x_target):
        # encode context and target
        z_context = self.context_encoder(x_context)
        with torch.no_grad():
            z_target = self.target_encoder(x_target)

        # predict target latents from context latents
        z_pred = self.predictor(z_context)
        return z_context, z_target, z_pred
