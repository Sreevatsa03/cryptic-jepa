import torch


def energy_loss(z_pred, z_target):
    """
    Compute the mean squared error between predicted and target latent representations
    """
    diff = z_pred - z_target
    return diff.pow(2).mean()


def variance_hinge_loss(z, gamma=1.0, eps=1e-4):
    """
    Loss to encourage each latent dimension to have variance above a certain threshold (gamma)
    This prevents collapse to constant features and encourages the model to use the full latent space
    """
    if z.ndim != 2:
        z = z.view(z.shape[0], -1)

    var = z.var(dim=0, unbiased=False)
    std = torch.sqrt(var + eps)
    return torch.relu(gamma - std).mean()


def covariance_loss(z):
    """
    Loss to encourage the latent representations to have low covariance
    This prevents the model from learning redundant features
    """
    if z.ndim != 2:
        z = z.view(z.shape[0], -1)

    batch_size, latent_dim = z.shape
    if batch_size < 2:
        return torch.zeros((), device=z.device)

    z = z - z.mean(dim=0)
    cov_z = (z.T @ z) / (batch_size - 1)
    mask = ~torch.eye(latent_dim, dtype=torch.bool, device=z.device)
    return cov_z[mask].pow(2).sum() / latent_dim


def jepa_loss(
    z_context,
    z_target,
    z_pred,
    apply_reg=False,
    reg_weight=1.0,
    reg_gamma=1.0,
):
    """
    Compute the total loss for JEPA training, which includes the energy loss and optional regularization
    """
    energy = energy_loss(z_pred, z_target)
    reg_loss = None
    if apply_reg:
        var_loss = variance_hinge_loss(z_context, gamma=reg_gamma)
        cov_loss = covariance_loss(z_context)
        
        # weight variance and covariance losses equally and combine
        reg_loss = var_loss + cov_loss
        energy = energy + reg_weight * reg_loss
    return energy, reg_loss
