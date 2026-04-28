import argparse
import os

import torch

from src.data.dataloader import create_dataloaders
from src.models.jepa_model import JEPAModel
from src.training.loss import energy_loss, jepa_loss


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_epoch(
    model,
    loader,
    optimizer,
    device,
    momentum,
    reg_weight,
    reg_gamma,
):
    """
    Train the model for one epoch and return average loss and regularization loss
    """
    model.train()
    total_loss = 0.0
    total_reg = 0.0
    total_samples = 0

    for context, target in loader:
        # move data to device
        context = context.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # forward pass, compute loss, and backpropagate
        optimizer.zero_grad()
        z_context, z_target, z_pred = model(context, target)
        loss, reg_loss = jepa_loss(
            z_context,
            z_target,
            z_pred,
            apply_reg=True,
            reg_weight=reg_weight,
            reg_gamma=reg_gamma,
        )
        loss.backward()
        optimizer.step()
        model.update_target_network(momentum=momentum)

        # accumulate total loss and regularization for monitoring
        batch_size = context.shape[0]
        total_loss += loss.item() * batch_size
        total_reg += (reg_loss.item() if reg_loss is not None else 0.0) * batch_size
        total_samples += batch_size

    # compute average loss and regularization over the epoch
    avg_loss = total_loss / max(total_samples, 1)
    avg_reg = total_reg / max(total_samples, 1)
    return avg_loss, avg_reg


@torch.no_grad()
def validate_epoch(model, loader, device):
    """
    Validate the model for one epoch and return average energy
    """
    model.eval()
    total_energy = 0.0
    total_samples = 0

    # iterate over validation data without computing gradients
    for context, target in loader:
        context = context.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        z_context, z_target, z_pred = model(context, target)
        energy = energy_loss(z_pred, z_target)
        batch_size = context.shape[0]
        total_energy += energy.item() * batch_size
        total_samples += batch_size

    return total_energy / max(total_samples, 1)


def main():
    parser = argparse.ArgumentParser(description="train eb-jepa on equilibrium data")
    parser.add_argument(
        "--contact-map-path",
        default="data/eq_jepa_contact_maps.pt",
        help="path to equilibrium contact map tensor",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--max-gap", type=int, default=50)
    parser.add_argument("--jitter-std", type=float, default=0.01)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--reg-weight", type=float, default=1.0)
    parser.add_argument("--reg-gamma", type=float, default=1.0)
    args = parser.parse_args()

    device = get_device()

    torch.manual_seed(args.seed)

    # create dataloaders for training and validation
    train_loader, val_loader = create_dataloaders(
        contact_map_path=args.contact_map_path,
        batch_size=32,
        max_gap=args.max_gap,
        jitter_std=args.jitter_std,
        val_split=args.val_split,
        seed=args.seed,
        pin_memory=False, # unnecessary for mps
        drop_last=True,
    )

    # initialize the model and optimizer
    model = JEPAModel(in_channels=1, latent_dim=128, predictor_hidden_dim=256)
    model = model.to(device)

    # only the context encoder and predictor are updated with gradients
    trainable_params = list(model.context_encoder.parameters()) + list(
        model.predictor.parameters()
    )
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=1e-4,
        weight_decay=1e-5,
    )

    # checkpointing setup to save the best model based on validation energy
    best_val = float("inf")
    checkpoint_dir = "models"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_jepa.pth")

    # training loop over epochs with monitoring of training loss, regularization, and validation energy
    for epoch in range(1, args.epochs + 1):
        train_loss, train_reg = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            momentum=args.momentum,
            reg_weight=args.reg_weight,
            reg_gamma=args.reg_gamma,
        )
        val_energy = validate_epoch(model, val_loader, device)

        if val_energy < best_val:
            best_val = val_energy
            torch.save(model.state_dict(), checkpoint_path)

        print(
            "Epoch {epoch:03d} | train_loss={train_loss:.6f} | "
            "train_reg={train_reg:.6f} | val_energy={val_energy:.6f}".format(
                epoch=epoch,
                train_loss=train_loss,
                train_reg=train_reg,
                val_energy=val_energy,
            )
        )


if __name__ == "__main__":
    main()
