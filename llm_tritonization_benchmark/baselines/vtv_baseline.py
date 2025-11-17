import torch

def vtv_pytorch(a, b):
    """
    PyTorch implementation of TSVC vtv function.
    
    Original C code:
    for (int nl = 0; nl < iterations*10; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[i] *= b[i];
        }
    }
    
    Arrays used: a (rw), b (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    
    a *= b
    
    return a