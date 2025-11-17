import torch

def s119_pytorch(aa, bb):
    """
    TSVC function s119 - 2D array dependency analysis
    
    Original C code:
    for (int nl = 0; nl < 200*(iterations/(LEN_2D)); nl++) {
        for (int i = 1; i < LEN_2D; i++) {
            for (int j = 1; j < LEN_2D; j++) {
                aa[i][j] = aa[i-1][j-1] + bb[i][j];
            }
        }
    }
    
    Arrays used: aa (rw), bb (r)
    """
    aa = aa.contiguous()
    bb = bb.contiguous()
    
    for i in range(1, aa.shape[0]):
        for j in range(1, aa.shape[1]):
            aa[i, j] = aa[i-1, j-1] + bb[i, j]
    
    return aa