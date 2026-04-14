import torch
from torch.utils.data import DataLoader
from src.data.dataset import train_dataset, test_dataset

# Generator must be on CPU for DataLoader shuffling, even when using CUDA
# This prevents "Expected 'cuda' device type for generator but found 'cpu'" error
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

generator = torch.Generator(device=DEVICE)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0,
    generator=generator
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)