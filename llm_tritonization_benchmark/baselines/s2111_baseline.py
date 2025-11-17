import torch

def s2111_pytorch(aa):
    """
    PyTorch implementation of TSVC s2111
    
    Original C code:
    for (int nl = 0; nl < 100*(iterations/(LEN_2D)); nl++) {
        for (int j = 1; j < LEN_2D; j++) {
            for (int i = 1; i < LEN_2D; i++) {
                aa[j][i] = (aa[j][i-1] + aa[j-1][i])/1.9;
            }
        }
    }
    
    Arrays used: aa (rw)
    """
    aa = aa.contiguous()
    
    for j in range(1, aa.shape[0]):
        for i in range(1, aa.shape[1]):
            aa[j, i] = (aa[j, i-1] + aa[j-1, i]) / 1.9
    
    return aa