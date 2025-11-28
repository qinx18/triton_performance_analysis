import torch

def s123_pytorch(a, b, c, d, e):
    """
    PyTorch implementation of TSVC s123 kernel.
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        j = -1;
        for (int i = 0; i < (LEN_1D/2); i++) {
            j++;
            a[j] = b[i] + d[i] * e[i];
            if (c[i] > (real_t)0.) {
                j++;
                a[j] = c[i] + d[i] * e[i];
            }
        }
    }
    
    Arrays: a (rw), b (r), c (r), d (r), e (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()
    
    len_half = len(b) // 2
    j = 0
    
    for i in range(len_half):
        # Always write first value
        a[j] = b[i] + d[i] * e[i]
        j += 1
        
        # Conditionally write second value
        if c[i] > 0.0:
            a[j] = c[i] + d[i] * e[i]
            j += 1