import torch

def s281_pytorch(a, b, c):
    """
    PyTorch implementation of TSVC s281 kernel.
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            x = a[LEN_1D-i-1] + b[i] * c[i];
            a[i] = x-(real_t)1.0;
            b[i] = x;
        }
    }
    
    Arrays: a (rw), b (rw), c (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    
    LEN_1D = a.size(0)
    
    for i in range(LEN_1D):
        x = a[LEN_1D - i - 1] + b[i] * c[i]
        a[i] = x - 1.0
        b[i] = x