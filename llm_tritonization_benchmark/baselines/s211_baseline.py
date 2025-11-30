import torch

def s211_pytorch(a, b, c, d, e):
    """
    PyTorch implementation of TSVC s211 kernel.
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 1; i < LEN_1D-1; i++) {
            a[i] = b[i - 1] + c[i] * d[i];
            b[i] = b[i + 1] - e[i] * d[i];
        }
    }
    
    Arrays: a (rw), b (rw), c (r), d (r), e (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()
    
    # Store b[i+1] values before modification
    b_next = b[2:].clone()
    
    # Update a[1:LEN_1D-1] and b[1:LEN_1D-1]
    a[1:-1] = b[:-2] + c[1:-1] * d[1:-1]
    b[1:-1] = b_next - e[1:-1] * d[1:-1]