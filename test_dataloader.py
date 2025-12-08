import torch
from torch.utils.data import DataLoader
import os
import multiprocessing

# Print system information
print(f"PyTorch version: {torch.__version__}")
print(f"Number of CPUs: {os.cpu_count()}")
print(f"Default multiprocessing start method: {multiprocessing.get_start_method(allow_none=True)}")

# Try different start methods
for method in ['fork', 'spawn', 'forkserver']:
    try:
        print(f"\nTrying multiprocessing start method: {method}")
        if multiprocessing.get_start_method(allow_none=True) != method:
            try:
                multiprocessing.set_start_method(method, force=True)
                print(f"Successfully set start method to {method}")
            except RuntimeError as e:
                print(f"Failed to set start method to {method}: {e}")
    except Exception as e:
        print(f"Error with {method}: {e}")

# Create a simple dataset
from torch.utils.data import Dataset
class SimpleDataset(Dataset):
    def __init__(self, size=1000):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.randn(3, 128, 128), torch.randn(5)

# Test with different num_workers
# Check if MPS is available
use_pin_memory = not (hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
print(f"Using pin_memory: {use_pin_memory}")

# When using MPS, only test with num_workers=0 to avoid multiprocessing issues
if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    worker_options = [0]
    print("⚠️ Testing only with num_workers=0 as multiprocessing can cause issues with MPS devices")
else:
    worker_options = [0, 1, 2, 4]

for num_workers in worker_options:
    try:
        print(f"\nTesting DataLoader with num_workers={num_workers}")
        dataset = SimpleDataset()
        loader = DataLoader(
            dataset, 
            batch_size=64, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=False if num_workers == 0 else True
        )

        # Try to iterate through the dataloader
        for i, (images, targets) in enumerate(loader):
            if i == 0:
                print(f"Successfully loaded batch with shape: {images.shape}")
            if i >= 2:  # Just test a few batches
                break
        print(f"DataLoader with num_workers={num_workers} works correctly")
    except Exception as e:
        print(f"Error with num_workers={num_workers}: {e}")
