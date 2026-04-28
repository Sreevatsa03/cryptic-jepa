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

    def forward(
        self,
        x_context,
        x_target,
        apply_reg=False,
        reg_weight=1.0,
        reg_gamma=1.0,
        return_reg=False,
    ):
        # encode context and target
        z_context = self.context_encoder(x_context)
        with torch.no_grad():
            z_target = self.target_encoder(x_target)

        # predict target latents from context latents
        z_pred = self.predictor(z_context)

        # compute energy as mean squared error between predicted and target latents
        diff = z_pred - z_target
        energy = diff.pow(2).sum(dim=1).mean()

        # compute regularization loss
        reg_loss = None
        if apply_reg:
            reg_loss = self._variance_hinge_loss(z_context, gamma=reg_gamma)
            energy = energy + reg_weight * reg_loss
        
        # return the regularization loss for monitoring
        if return_reg:
            return energy, reg_loss
        return energy

    @staticmethod
    def _variance_hinge_loss(z, gamma=1.0, eps=1e-4):
        """
        VICReg-style variance hinge loss to encourage feature diversity in the latent space
        """
        if z.ndim != 2:
            z = z.view(z.shape[0], -1)

        var = z.var(dim=0, unbiased=False)
        std = torch.sqrt(var + eps)
        return torch.relu(gamma - std).mean()
