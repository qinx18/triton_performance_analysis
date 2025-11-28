import torch

def s1221_pytorch(a, b):
    """
    PyTorch implementation of TSVC s1221 kernel.
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 4; i < LEN_1D; i++) {
            b[i] = b[i - 4] + a[i];
        }
    }
    
    Arrays: a (read), b (read-write)
    """
    a = a.contiguous()
    b = b.contiguous()
    
    b[4:] = b[:-4] + a[4:]