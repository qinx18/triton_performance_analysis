import torch

def s293_pytorch(a):
    """
    TSVC s293 - Array assignment with scalar broadcast
    
    Original C code:
    for (int nl = 0; nl < 4*iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[i] = a[0];
        }
    }
    
    Arrays: a (rw)
    """
    a = a.contiguous()
    
    # Broadcast a[0] to all elements
    a[:] = a[0]