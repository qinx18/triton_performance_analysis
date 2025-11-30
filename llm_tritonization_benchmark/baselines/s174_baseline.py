import torch

def s174_pytorch(a, b, m):
    """
    PyTorch implementation of TSVC s174
    
    Original C code:
    for (int nl = 0; nl < 10*iterations; nl++) {
        for (int i = 0; i < M; i++) {
            a[i+M] = a[i] + b[i];
        }
    }
    
    Arrays: a (rw), b (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    
    a[m:2*m] = a[:m] + b[:m]