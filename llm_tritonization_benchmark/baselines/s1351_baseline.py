import torch

def s1351_pytorch(a, b, c):
    """
    TSVC s1351 - Vector addition with pointer arithmetic
    
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
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    
    a[:] = b + c