import torch

def s132_pytorch(aa, b, c, j, k):
    """
    PyTorch implementation of TSVC s132
    
    Original C code:
    for (int nl = 0; nl < 400*iterations; nl++) {
        for (int i= 1; i < LEN_2D; i++) {
            aa[j][i] = aa[k][i-1] + b[i] * c[1];
        }
    }
    
    Arrays: aa (rw), b (r), c (r)
    Scalars: j, k
    """
    aa = aa.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    
    LEN_2D = aa.shape[1]
    
    for i in range(1, LEN_2D):
        aa[j, i] = aa[k, i-1] + b[i] * c[1]