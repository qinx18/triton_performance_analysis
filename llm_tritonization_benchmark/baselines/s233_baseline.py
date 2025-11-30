import torch

def s233_pytorch(aa, bb, cc):
    """
    PyTorch implementation of TSVC s233 kernel.
    
    Original C code:
    for (int nl = 0; nl < 100*(iterations/LEN_2D); nl++) {
        for (int i = 1; i < LEN_2D; i++) {
            for (int j = 1; j < LEN_2D; j++) {
                aa[j][i] = aa[j-1][i] + cc[j][i];
            }
            for (int j = 1; j < LEN_2D; j++) {
                bb[j][i] = bb[j][i-1] + cc[j][i];
            }
        }
    }
    
    Arrays: aa (rw), bb (rw), cc (r)
    """
    aa = aa.contiguous()
    bb = bb.contiguous()
    cc = cc.contiguous()
    
    LEN_2D = aa.shape[0]
    
    for i in range(1, LEN_2D):
        for j in range(1, LEN_2D):
            aa[j, i] = aa[j-1, i] + cc[j, i]
        for j in range(1, LEN_2D):
            bb[j, i] = bb[j, i-1] + cc[j, i]