import torch

def s113_pytorch(a, b):
    """
    PyTorch implementation of TSVC s113
    
    Original C code:
    for (int nl = 0; nl < 4*iterations; nl++) {
        for (int i = 1; i < LEN_1D; i++) {
            a[i] = a[0] + b[i];
        }
    }
    
    Arrays: a (rw), b (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    
    a[1:] = a[0] + b[1:]