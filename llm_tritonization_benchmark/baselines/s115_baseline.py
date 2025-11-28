import torch

def s115_pytorch(a, aa):
    """
    PyTorch implementation of TSVC s115 kernel.
    
    Original C code:
    for (int nl = 0; nl < 1000*(iterations/LEN_2D); nl++) {
        for (int j = 0; j < LEN_2D; j++) {
            for (int i = j+1; i < LEN_2D; i++) {
                a[i] -= aa[j][i] * a[j];
            }
        }
    }
    
    Arrays: a (rw), aa (r)
    """
    a = a.contiguous()
    aa = aa.contiguous()
    
    LEN_2D = aa.size(0)
    
    for j in range(LEN_2D):
        for i in range(j+1, LEN_2D):
            a[i] -= aa[j, i] * a[j]