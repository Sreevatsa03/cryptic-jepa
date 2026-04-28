from torch import nn


class LatentPredictor(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=256):
        super().__init__()
        
        self.net = nn.Sequential(
            # expand the latent representation
            nn.Linear(latent_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            
            # project back to the target latent dimension
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x):
        return self.net(x)