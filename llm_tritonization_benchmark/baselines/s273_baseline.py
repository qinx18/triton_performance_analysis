import torch

def s273_pytorch(a, b, c, d, e):
    """
    PyTorch implementation of TSVC s273.
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[i] += d[i] * e[i];
            if (a[i] < (real_t)0.)
                b[i] += d[i] * e[i];
            c[i] += a[i] * d[i];
        }
    }
    
    Arrays: a (rw), b (rw), c (rw), d (r), e (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()
    
    # a[i] += d[i] * e[i]
    de_product = d * e
    a += de_product
    
    # if (a[i] < 0.) b[i] += d[i] * e[i]
    mask = a < 0.0
    b += torch.where(mask, de_product, 0.0)
    
    # c[i] += a[i] * d[i]
    c += a * d
    
    return a, b, c