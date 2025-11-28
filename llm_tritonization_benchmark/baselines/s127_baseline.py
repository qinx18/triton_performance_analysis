import torch

def s127_pytorch(a, b, c, d, e):
    """
    PyTorch implementation of TSVC s127 kernel.
    
    Original C code:
    for (int nl = 0; nl < 2*iterations; nl++) {
        j = -1;
        for (int i = 0; i < LEN_1D/2; i++) {
            j++;
            a[j] = b[i] + c[i] * d[i];
            j++;
            a[j] = b[i] + d[i] * e[i];
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
    
    for i in range(len_half):
        j = 2 * i
        a[j] = b[i] + c[i] * d[i]
        a[j + 1] = b[i] + d[i] * e[i]