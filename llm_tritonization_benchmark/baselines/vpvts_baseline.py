import torch

def vpvts_pytorch(a, b, s):
    """
    PyTorch implementation of TSVC vpvts function.
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[i] += b[i] * s;
        }
    }
    
    Arrays used: a (rw), b (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    
    a += b * s
    
    return a