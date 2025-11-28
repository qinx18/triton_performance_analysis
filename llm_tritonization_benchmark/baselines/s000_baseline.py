import torch

def s000_pytorch(a, b):
    """
    PyTorch implementation of TSVC s000 kernel.
    
    Original C code:
    for (int nl = 0; nl < 2*iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[i] = b[i] + 1;
        }
    }
    
    Arrays: a (rw), b (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    
    a[:] = b + 1