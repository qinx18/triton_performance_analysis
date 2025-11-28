import torch

def s1232_pytorch(aa, bb, cc):
    """
    PyTorch implementation of TSVC s1232 kernel.
    
    Original C code:
    for (int nl = 0; nl < 100*(iterations/LEN_2D); nl++) {
        for (int j = 0; j < LEN_2D; j++) {
            for (int i = j; i < LEN_2D; i++) {
                aa[i][j] = bb[i][j] + cc[i][j];
            }
        }
    }
    
    Arrays: aa (write), bb (read), cc (read)
    """
    aa = aa.contiguous()
    bb = bb.contiguous()
    cc = cc.contiguous()
    
    len_2d = aa.size(0)
    
    for j in range(len_2d):
        for i in range(j, len_2d):
            aa[i, j] = bb[i, j] + cc[i, j]