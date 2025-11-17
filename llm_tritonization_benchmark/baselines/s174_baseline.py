import torch

def s174_pytorch(a, b):
    """
    PyTorch implementation of TSVC s174.
    
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
    
    M = b.size(0)
    
    # a[i+M] = a[i] + b[i] for i in range(M)
    a[M:2*M] = a[:M] + b
    
    return a