import torch

def s261_pytorch(a, b, c, d):
    """
    PyTorch implementation of TSVC s261.
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 1; i < LEN_1D; ++i) {
            t = a[i] + b[i];
            a[i] = t + c[i-1];
            t = c[i] * d[i];
            c[i] = t;
        }
    }
    
    Arrays: a (rw), b (r), c (rw), d (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    
    for i in range(1, len(a)):
        t = a[i] + b[i]
        a[i] = t + c[i-1]
        t = c[i] * d[i]
        c[i] = t
    
    return a, c