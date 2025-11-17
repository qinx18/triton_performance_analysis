import torch

def s313_pytorch(a, b):
    """
    PyTorch implementation of TSVC s313 - dot product computation
    
    Original C code:
    for (int nl = 0; nl < iterations*5; nl++) {
        dot = (real_t)0.;
        for (int i = 0; i < LEN_1D; i++) {
            dot += a[i] * b[i];
        }
    }
    
    Arrays used: a (r), b (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    
    dot = torch.sum(a * b)
    
    return a, b