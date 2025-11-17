import torch

def s173_pytorch(a, b, k):
    """
    PyTorch implementation of TSVC s173
    
    Original C code:
    for (int nl = 0; nl < 10*iterations; nl++) {
        for (int i = 0; i < LEN_1D/2; i++) {
            a[i+k] = a[i] + b[i];
        }
    }
    
    Arrays: a (rw), b (r)
    Scalar parameters: k
    """
    a = a.contiguous()
    b = b.contiguous()
    
    len_1d = a.size(0)
    half_len = len_1d // 2
    
    for i in range(half_len):
        if i + k < len_1d:
            a[i + k] = a[i] + b[i]
    
    return a