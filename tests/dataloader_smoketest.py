from pathlib import Path

from src.data.dataloader import create_dataloaders


def main():
    contact_map_path = Path("data/eq_jepa_contact_maps.pt")
    if not contact_map_path.exists():
        print(f"missing {contact_map_path}; run data_processing first")
        return

    train_loader, val_loader = create_dataloaders(
        contact_map_path=str(contact_map_path),
        batch_size=4,
        max_gap=50,
        jitter_std=0.01,
        val_split=0.1,
        seed=42,
        num_workers=0,
    )

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    print("train batch shapes:", train_batch[0].shape, train_batch[1].shape)
    print("val batch shapes:", val_batch[0].shape, val_batch[1].shape)


if __name__ == "__main__":
    main()
