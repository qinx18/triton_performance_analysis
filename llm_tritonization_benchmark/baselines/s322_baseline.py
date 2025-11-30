import torch

def s322_pytorch(a, b, c):
    """
    PyTorch implementation of TSVC s322 kernel.
    
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
    
    a[2:] = a[2:] + a[1:-1] * b[2:] + a[:-2] * c[2:]