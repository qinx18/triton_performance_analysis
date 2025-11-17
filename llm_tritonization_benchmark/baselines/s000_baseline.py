import torch

def s000_pytorch(a, b, iterations):
    """
    TSVC s000 - Simple vector addition with scalar
    
    Original C code:
    for (int nl = 0; nl < 2*iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[i] = b[i] + 1;
        }
    }
    
    Arrays: a (rw), b (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    
    a[:] = b + 1
    
    return a