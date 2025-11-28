import torch

def s1115_pytorch(aa, bb, cc):
    """
    PyTorch implementation of TSVC s1115
    
    Original C code:
    for (int nl = 0; nl < 100*(iterations/LEN_2D); nl++) {
        for (int i = 0; i < LEN_2D; i++) {
            for (int j = 0; j < LEN_2D; j++) {
                aa[i][j] = aa[i][j]*cc[j][i] + bb[i][j];
            }
        }
    }
    
    Arrays: aa (rw), bb (r), cc (r)
    """
    aa = aa.contiguous()
    bb = bb.contiguous()
    cc = cc.contiguous()
    
    aa[:] = aa * cc.T + bb