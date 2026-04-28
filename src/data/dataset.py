import torch
from torch.utils.data import Dataset


class ContactMapPairDataset(Dataset):
    """
    Dataset that samples context-target pairs from contact maps with an optional max gap
    """

    def __init__(
        self,
        contact_map_path=None,
        contact_maps=None,
        max_gap=50,
        jitter_std=0.0,
    ):
        if contact_maps is None:
            if not contact_map_path:
                raise ValueError("contact_map_path or contact_maps must be provided")
            contact_maps = torch.load(contact_map_path, map_location="cpu")

        if contact_maps.dim() == 3:
            contact_maps = contact_maps.unsqueeze(1)
        if contact_maps.dim() != 4:
            raise ValueError("contact_maps must have shape (frames, 1, atoms, atoms)")

        if contact_maps.shape[0] == 0:
            raise ValueError("contact_maps contains no frames")

        self.contact_maps = contact_maps.float()
        self.n_frames = contact_maps.shape[0]
        self.max_gap = max(0, int(max_gap))
        self.max_gap = min(self.max_gap, self.n_frames - 1)
        self.jitter_std = float(jitter_std)

    def __len__(self):
        return self.n_frames

    def _sample_target_index(self, idx):
        if self.n_frames == 1 or self.max_gap == 0:
            return idx

        low = max(0, idx - self.max_gap)
        high = min(self.n_frames - 1, idx + self.max_gap)

        if low == high:
            return low

        target_idx = idx
        while target_idx == idx:
            target_idx = torch.randint(low, high + 1, (1,)).item()
        return target_idx

    def __getitem__(self, idx):
        context = self.contact_maps[idx]
        target_idx = self._sample_target_index(idx)
        target = self.contact_maps[target_idx]

        if self.jitter_std > 0:
            context = context + torch.randn_like(context) * self.jitter_std
            target = target + torch.randn_like(target) * self.jitter_std

        return context, target
