import torch

def s114_pytorch(aa, bb):
    """
    PyTorch implementation of TSVC s114 function.
    
    Original C code:
    for (int nl = 0; nl < 200*(iterations/(LEN_2D)); nl++) {
        for (int i = 0; i < LEN_2D; i++) {
            for (int j = 0; j < i; j++) {
                aa[i][j] = aa[j][i] + bb[i][j];
            }
        }
    }
    
    Arrays: aa (rw), bb (r)
    """
    aa = aa.contiguous()
    bb = bb.contiguous()
    
    len_2d = aa.size(0)
    
    for i in range(len_2d):
        for j in range(i):
            aa[i, j] = aa[j, i] + bb[i, j]