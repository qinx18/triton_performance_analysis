import torch

def s1351_pytorch(a, b, c):
    """
    PyTorch implementation of TSVC s1351 - basic linear combination.
    
    Original C code:
    for (int nl = 0; nl < 8*iterations; nl++) {
        real_t* __restrict__ A = a;
        real_t* __restrict__ B = b;
        real_t* __restrict__ C = c;
        for (int i = 0; i < LEN_1D; i++) {
            *A = *B+*C;
            A++;
            B++;
            C++;
        }
    }
    
    Arrays: a (w), b (r), c (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    
    a[:] = b + c
    
    return a