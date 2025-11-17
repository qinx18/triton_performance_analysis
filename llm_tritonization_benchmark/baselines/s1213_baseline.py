import torch

def s1213_pytorch(a, b, c, d):
    """
    PyTorch implementation of TSVC s1213
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 1; i < LEN_1D-1; i++) {
            a[i] = b[i-1]+c[i];
            b[i] = a[i+1]*d[i];
        }
    }
    
    Arrays: a (rw), b (rw), c (r), d (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    
    # Store original a[i+1] values before they get modified
    a_next = a[2:].clone()
    
    # Update arrays for i in range [1, LEN_1D-1)
    a[1:-1] = b[:-2] + c[1:-1]
    b[1:-1] = a_next * d[1:-1]
    
    return a, b