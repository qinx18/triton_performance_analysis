import torch

def s128_pytorch(a, b, c, d):
    """
    PyTorch implementation of TSVC s128.
    
    Original C code:
    for (int nl = 0; nl < 2*iterations; nl++) {
        j = -1;
        for (int i = 0; i < LEN_1D/2; i++) {
            k = j + 1;
            a[i] = b[k] - d[i];
            j = k + 1;
            b[k] = a[i] + c[k];
        }
    }
    
    Arrays: a (rw), b (rw), c (r), d (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    
    len_1d = len(a)
    half_len = len_1d // 2
    
    j = -1
    for i in range(half_len):
        k = j + 1
        a[i] = b[k] - d[i]
        j = k + 1
        b[k] = a[i] + c[k]
    
    return a, b