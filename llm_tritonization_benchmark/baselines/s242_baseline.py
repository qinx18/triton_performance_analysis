import torch

def s242_pytorch(a, b, c, d, s1, s2):
    """
    PyTorch implementation of TSVC s242 function.
    
    Original C code:
    for (int nl = 0; nl < iterations/5; nl++) {
        for (int i = 1; i < LEN_1D; ++i) {
            a[i] = a[i - 1] + s1 + s2 + b[i] + c[i] + d[i];
        }
    }
    
    Arrays: a (rw), b (r), c (r), d (r)
    Scalars: s1, s2
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    
    for i in range(1, len(a)):
        a[i] = a[i - 1] + s1 + s2 + b[i] + c[i] + d[i]