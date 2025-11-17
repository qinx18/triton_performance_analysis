import torch

def s293_pytorch(a):
    """
    PyTorch implementation of TSVC s293 function.
    
    Original C code:
    for (int nl = 0; nl < 4*iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[i] = a[0];
        }
    }
    
    Arrays used: a (rw)
    """
    a = a.contiguous()
    a[:] = a[0]
    return a