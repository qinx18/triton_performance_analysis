import torch

def s111_pytorch(a, b):
    """
    PyTorch implementation of TSVC s111 function.
    
    Original C loop code:
    for (int nl = 0; nl < 2*iterations; nl++) {
        for (int i = 1; i < LEN_1D; i += 2) {
            a[i] = a[i - 1] + b[i];
        }
    }
    
    Arrays used: a (rw), b (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    
    # for (int i = 1; i < LEN_1D; i += 2)
    indices = torch.arange(1, a.size(0), 2, device=a.device)
    
    if indices.numel() > 0:
        a[indices] = a[indices - 1] + b[indices]