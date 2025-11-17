import torch

def s1421_pytorch(a, b, xx):
    """
    PyTorch implementation of TSVC s1421.
    
    Original C code:
    for (int nl = 0; nl < 8*iterations; nl++) {
        for (int i = 0; i < LEN_1D/2; i++) {
            b[i] = xx[i] + a[i];
        }
    }
    
    Arrays: a (r), b (rw), xx (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    xx = xx.contiguous()
    
    len_1d = a.size(0)
    half_len = len_1d // 2
    
    b[:half_len] = xx[:half_len] + a[:half_len]
    
    return b