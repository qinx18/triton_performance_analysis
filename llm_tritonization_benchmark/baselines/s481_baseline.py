import torch

def s481_pytorch(a, b, c, d):
    """
    PyTorch implementation of TSVC s481.
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            if (d[i] < (real_t)0.) {
                exit (0);
            }
            a[i] += b[i] * c[i];
        }
    }
    
    Arrays: a (rw), b (r), c (r), d (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    
    # Check if any element in d is negative (equivalent to exit condition)
    if torch.any(d < 0.0):
        # In PyTorch, we can't exit, so we'll raise an exception or return early
        # For functional equivalence, we'll just return the original array
        return a
    
    # Perform the computation: a[i] += b[i] * c[i]
    a = a + b * c
    
    return a