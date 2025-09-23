import os
import torch
import numpy as np

def set_seed(seed=42):
    """
    Sets the seed for reproducibility for numpy, torch, and python's hash function.
    Ensures that operations are deterministic on GPU as well.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_device():
    """
    Checks for GPU availability and sets up the master device for model and data placement.
    """
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")
        torch.cuda.set_device(DEVICE)
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        DEVICE = torch.device("cpu")
        print("CUDA is not available. Using CPU.")
    return DEVICE