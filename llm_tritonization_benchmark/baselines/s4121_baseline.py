import torch

def s4121_pytorch(a, b, c):
    """
    PyTorch implementation of TSVC s4121.
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[i] += f(b[i],c[i]);
        }
    }
    
    Arrays: a (rw), b (r), c (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    
    # f(b[i],c[i]) is typically implemented as b[i] * c[i] in TSVC
    a += b * c
    
    return a