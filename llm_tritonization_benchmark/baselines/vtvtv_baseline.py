import torch

def vtvtv_pytorch(a, b, c):
    """
    PyTorch implementation of TSVC vtvtv function.
    
    Original C code:
    for (int nl = 0; nl < 4*iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[i] = a[i] * b[i] * c[i];
        }
    }
    
    Arrays used: a (rw), b (r), c (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    
    a[:] = a * b * c