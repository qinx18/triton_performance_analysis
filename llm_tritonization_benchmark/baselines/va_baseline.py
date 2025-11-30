import torch

def va_pytorch(a, b):
    """
    TSVC function va - simple array copy operation.
    
    Original C code:
    for (int nl = 0; nl < iterations*10; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[i] = b[i];
        }
    }
    
    Arrays: a (rw), b (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    
    a[:] = b