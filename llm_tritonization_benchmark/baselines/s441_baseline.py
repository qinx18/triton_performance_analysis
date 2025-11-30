import torch

def s441_pytorch(a, b, c, d):
    """
    PyTorch implementation of TSVC s441 - conditional assignments with three branches
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            if (d[i] < (real_t)0.) {
                a[i] += b[i] * c[i];
            } else if (d[i] == (real_t)0.) {
                a[i] += b[i] * b[i];
            } else {
                a[i] += c[i] * c[i];
            }
        }
    }
    
    Args:
        a: read-write tensor
        b: read tensor
        c: read tensor  
        d: read-write tensor
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    
    # Three-way conditional using nested torch.where
    result = torch.where(d < 0.0, 
                        a + b * c,
                        torch.where(d == 0.0,
                                   a + b * b,
                                   a + c * c))
    
    a[:] = result