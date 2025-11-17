import torch

def s162_pytorch(a, b, c, k):
    """
    PyTorch implementation of TSVC s162.
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        if (k > 0) {
            for (int i = 0; i < LEN_1D-1; i++) {
                a[i] = a[i + k] + b[i] * c[i];
            }
        }
    }
    
    Arrays: a (rw), b (r), c (r)
    Scalar parameters: k
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    
    if k > 0 and k < len(a):
        len_1d = len(a)
        for i in range(len_1d - 1):
            if i + k < len_1d:
                a[i] = a[i + k] + b[i] * c[i]
    
    return a