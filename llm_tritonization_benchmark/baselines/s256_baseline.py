import torch

def s256_pytorch(a, aa, bb, d):
    """
    PyTorch implementation of TSVC s256 kernel.
    
    Original C code:
    for (int nl = 0; nl < 10*(iterations/LEN_2D); nl++) {
        for (int i = 0; i < LEN_2D; i++) {
            for (int j = 1; j < LEN_2D; j++) {
                a[j] = (real_t)1.0 - a[j - 1];
                aa[j][i] = a[j] + bb[j][i]*d[j];
            }
        }
    }
    
    Arrays: a (rw), aa (w), bb (r), d (r)
    """
    a = a.contiguous()
    aa = aa.contiguous()
    bb = bb.contiguous()
    d = d.contiguous()
    
    LEN_2D = a.shape[0]
    
    for i in range(LEN_2D):
        for j in range(1, LEN_2D):
            a[j] = 1.0 - a[j - 1]
            aa[j, i] = a[j] + bb[j, i] * d[j]