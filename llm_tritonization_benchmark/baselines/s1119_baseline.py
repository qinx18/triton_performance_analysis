import torch

def s1119_pytorch(aa, bb):
    """
    PyTorch implementation of TSVC s1119
    
    Original C code:
    for (int nl = 0; nl < 200*(iterations/(LEN_2D)); nl++) {
        for (int i = 1; i < LEN_2D; i++) {
            for (int j = 0; j < LEN_2D; j++) {
                aa[i][j] = aa[i-1][j] + bb[i][j];
            }
        }
    }
    
    Arrays: aa (rw), bb (r)
    """
    aa = aa.contiguous()
    bb = bb.contiguous()
    
    LEN_2D = aa.shape[0]
    
    for i in range(1, LEN_2D):
        for j in range(LEN_2D):
            aa[i, j] = aa[i-1, j] + bb[i, j]