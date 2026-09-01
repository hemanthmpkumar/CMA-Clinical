import torch
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        # Example tensor generation
        return torch.randn(3, 224, 224)

def create_dataloader() -> DataLoader:
    dataset = CustomDataset()
    
    return DataLoader(
        dataset,
        batch_size=64,            # Large batch size to saturate 32 GPU cores
        num_workers=8,            # 8 workers match P-cores without OS saturation
        pin_memory=True,          # Enables fast host-to-device MPS memory copy
        persistent_workers=True,  # Reuses processes across iterations
        prefetch_factor=2,        # Preloads next 2 batches per worker
    )