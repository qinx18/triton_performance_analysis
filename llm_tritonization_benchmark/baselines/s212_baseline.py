import torch

def s212_pytorch(a, b, c, d):
    """
    PyTorch implementation of TSVC s212
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D-1; i++) {
            a[i] *= c[i];
            b[i] += a[i + 1] * d[i];
        }
    }
    
    Arrays: a (rw), b (rw), c (r), d (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    
    # First update a[i] *= c[i] for i in range(len-1)
    a[:-1] *= c[:-1]
    
    # Then update b[i] += a[i + 1] * d[i] for i in range(len-1)
    b[:-1] += a[1:] * d[:-1]
    
    return a, b