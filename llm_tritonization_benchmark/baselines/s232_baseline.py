import torch

def s232_pytorch(aa, bb):
    """
    PyTorch implementation of TSVC s232.
    
    Original C code:
    for (int nl = 0; nl < 100*(iterations/(LEN_2D)); nl++) {
        for (int j = 1; j < LEN_2D; j++) {
            for (int i = 1; i <= j; i++) {
                aa[j][i] = aa[j][i-1]*aa[j][i-1]+bb[j][i];
            }
        }
    }
    
    Arrays: aa (rw), bb (r)
    """
    aa = aa.contiguous()
    bb = bb.contiguous()
    
    LEN_2D = aa.shape[0]
    
    for j in range(1, LEN_2D):
        for i in range(1, j + 1):
            aa[j, i] = aa[j, i-1] * aa[j, i-1] + bb[j, i]
    
    return aa