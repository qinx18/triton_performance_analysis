import torch

def s322_pytorch(a, b, c):
    """
    PyTorch implementation of TSVC s322.
    
    Original C code:
    for (int nl = 0; nl < iterations/2; nl++) {
        for (int i = 2; i < LEN_1D; i++) {
            a[i] = a[i] + a[i - 1] * b[i] + a[i - 2] * c[i];
        }
    }
    
    Arrays: a (rw), b (r), c (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    
    for i in range(2, len(a)):
        a[i] = a[i] + a[i - 1] * b[i] + a[i - 2] * c[i]
    
    return a