import torch

def s321_pytorch(a, b):
    """
    PyTorch implementation of TSVC s321 kernel.
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 1; i < LEN_1D; i++) {
            a[i] += a[i-1] * b[i];
        }
    }
    
    Arrays: a (rw), b (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    
    for i in range(1, len(a)):
        a[i] += a[i-1] * b[i]