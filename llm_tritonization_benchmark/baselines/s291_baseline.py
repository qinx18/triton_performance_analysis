import torch

def s291_pytorch(a, b):
    """
    PyTorch implementation of TSVC s291.
    
    Original C code:
    for (int nl = 0; nl < 2*iterations; nl++) {
        im1 = LEN_1D-1;
        for (int i = 0; i < LEN_1D; i++) {
            a[i] = (b[i] + b[im1]) * (real_t).5;
            im1 = i;
        }
    }
    
    Arrays: a (rw), b (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    
    n = len(b)
    im1 = n - 1
    
    for i in range(n):
        a[i] = (b[i] + b[im1]) * 0.5
        im1 = i
    
    return a