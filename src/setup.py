import random
import numpy as np
import torch


RANDOM_SEED = 42


def setup_device_with_seeds(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cpu"
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0)/1024**3:.1f} GB")
        print(f"Memory cached: {torch.cuda.memory_reserved(0)/1024**3:.1f} GB")
        device = "cuda"
    elif torch.backends.mps.is_available():  # For Apple Silicon devices
        torch.mps.manual_seed(seed)
        device = "mps"

    print(f"Using device: {device}")
    return device
