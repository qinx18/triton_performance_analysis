import torch

def s278_pytorch(a, b, c, d, e):
    """
    PyTorch implementation of TSVC s278 kernel.
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            if (a[i] > (real_t)0.) {
                goto L20;
            }
            b[i] = -b[i] + d[i] * e[i];
            goto L30;
    L20:
            c[i] = -c[i] + d[i] * e[i];
    L30:
            a[i] = b[i] + c[i] * d[i];
        }
    }
    
    Arrays: a (rw), b (rw), c (rw), d (r), e (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()
    
    # Condition: a[i] > 0
    condition = a > 0.0
    
    # When condition is False: b[i] = -b[i] + d[i] * e[i]
    # When condition is True: c[i] = -c[i] + d[i] * e[i]
    de_product = d * e
    
    b[:] = torch.where(condition, b, -b + de_product)
    c[:] = torch.where(condition, -c + de_product, c)
    
    # Always: a[i] = b[i] + c[i] * d[i]
    a[:] = b + c * d