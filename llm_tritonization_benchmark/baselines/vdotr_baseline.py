import torch

def vdotr_pytorch(a, b):
    """
    PyTorch implementation of TSVC vdotr function.
    
    Original C code:
    for (int nl = 0; nl < iterations*10; nl++) {
        dot = 0.;
        for (int i = 0; i < LEN_1D; i++) {
            dot += a[i] * b[i];
        }
    }
    
    Arrays: a (read), b (read)
    """
    a = a.contiguous()
    b = b.contiguous()
    
    dot = torch.sum(a * b)
    
    return dot