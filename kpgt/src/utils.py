import os
import random
import numpy as np
import torch
import dgl

def set_random_seed(seed=22, n_threads=16):
    """Set random seed.

    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # DGL随机种子设置需要特殊处理，因为CUDA版本的DGL在CPU模式下可能会失败
    try:
        dgl.random.seed(seed)
        dgl.seed(seed)
    except Exception as e:
        print(f"Warning: Failed to set DGL random seed: {e}")
        print("This is normal when using CPU-only mode with CUDA-enabled DGL installation")
    
    torch.manual_seed(seed)
    
    # 只有在CUDA可用时才设置CUDA相关的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    torch.set_num_threads(n_threads)
    os.environ['PYTHONHASHSEED'] = str(seed) 