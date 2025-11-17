import torch

def s1232_pytorch(aa, bb, cc):
    """
    PyTorch implementation of TSVC s1232
    
    Original C code:
    for (int nl = 0; nl < 100*(iterations/LEN_2D); nl++) {
        for (int j = 0; j < LEN_2D; j++) {
            for (int i = j; i < LEN_2D; i++) {
                aa[i][j] = bb[i][j] + cc[i][j];
            }
        }
    }
    
    Arrays used: aa (r), bb (r), cc (r)
    """
    aa = aa.contiguous()
    bb = bb.contiguous()
    cc = cc.contiguous()
    
    LEN_2D = aa.shape[0]
    
    for j in range(LEN_2D):
        for i in range(j, LEN_2D):
            aa[i, j] = bb[i, j] + cc[i, j]
    
    return aa