import torch

def s231_pytorch(aa, bb):
    """
    PyTorch implementation of TSVC s231 kernel.
    
    Original C code:
    for (int nl = 0; nl < 100*(iterations/LEN_2D); nl++) {
        for (int i = 0; i < LEN_2D; ++i) {
            for (int j = 1; j < LEN_2D; j++) {
                aa[j][i] = aa[j - 1][i] + bb[j][i];
            }
        }
    }
    
    Arrays: aa (rw), bb (r)
    """
    aa = aa.contiguous()
    bb = bb.contiguous()
    
    LEN_2D = aa.shape[0]
    
    for i in range(LEN_2D):
        for j in range(1, LEN_2D):
            aa[j, i] = aa[j - 1, i] + bb[j, i]