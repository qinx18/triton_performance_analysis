import torch

def s241_pytorch(a, b, c, d):
    """
    PyTorch implementation of TSVC s241 kernel.
    
    Original C code:
    for (int nl = 0; nl < 2*iterations; nl++) {
        for (int i = 0; i < LEN_1D-1; i++) {
            a[i] = b[i] * c[i  ] * d[i];
            b[i] = a[i] * a[i+1] * d[i];
        }
    }
    
    Arrays: a (rw), b (rw), c (r), d (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    
    n = a.size(0) - 1

    # Save old a values - b[i] needs OLD a[i+1] (not yet computed in iteration i)
    a_old = a[:n+1].clone()

    # First statement: a[i] = b[i] * c[i] * d[i]
    a[:n] = b[:n] * c[:n] * d[:n]

    # Second statement: b[i] = a[i] * a[i+1] * d[i]
    # Uses NEW a[i] (just computed) and OLD a[i+1] (not yet computed)
    b[:n] = a[:n] * a_old[1:n+1] * d[:n]