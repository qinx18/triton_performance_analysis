import torch

def vpvts_pytorch(a, b, s):
    """
    PyTorch implementation of TSVC vpvts kernel.
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[i] += b[i] * s;
        }
    }
    
    Arrays: a (rw), b (r)
    Scalar parameters: s
    """
    a = a.contiguous()
    b = b.contiguous()
    
    a[:] += b * s