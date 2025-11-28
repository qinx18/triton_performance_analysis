import torch

def s122_pytorch(a, b, n1, n3):
    """
    PyTorch implementation of TSVC s122
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        j = 1;
        k = 0;
        for (int i = n1-1; i < LEN_1D; i += n3) {
            k += j;
            a[i] += b[LEN_1D - k];
        }
    }
    
    Arrays: a (rw), b (r)
    Scalar parameters: n1, n3
    """
    a = a.contiguous()
    b = b.contiguous()
    
    LEN_1D = a.shape[0]
    
    j = 1
    k = 0
    
    for i in range(n1 - 1, LEN_1D, n3):
        k += j
        a[i] += b[LEN_1D - k]