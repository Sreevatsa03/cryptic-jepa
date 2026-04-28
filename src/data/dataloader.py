import torch
from torch.utils.data import DataLoader, Subset

from .dataset import ContactMapPairDataset


def create_dataloaders(
    contact_map_path,
    batch_size=32,
    max_gap=50,
    jitter_std=0.01,
    val_split=0.1,
    seed=42,
    num_workers=0,
    pin_memory=False,
    drop_last=False,
):
    if not (0.0 < val_split < 1.0):
        raise ValueError("val_split must be between 0 and 1")

    contact_maps = torch.load(contact_map_path, map_location="cpu")
    if contact_maps.dim() == 3:
        contact_maps = contact_maps.unsqueeze(1)
    if contact_maps.dim() != 4:
        raise ValueError("contact_maps must have shape (frames, 1, atoms, atoms)")

    n_frames = contact_maps.shape[0]
    if n_frames < 2:
        raise ValueError("contact_maps must contain at least 2 frames")

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    indices = torch.randperm(n_frames, generator=generator).tolist()
    split_idx = int(n_frames * (1.0 - val_split))
    if split_idx == 0 or split_idx == n_frames:
        raise ValueError("val_split leaves no samples for train or val")

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_dataset = ContactMapPairDataset(
        contact_maps=contact_maps,
        max_gap=max_gap,
        jitter_std=jitter_std,
    )
    val_dataset = ContactMapPairDataset(
        contact_maps=contact_maps,
        max_gap=max_gap,
        jitter_std=0.0,
    )

    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    val_loader = DataLoader(
        Subset(val_dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader
