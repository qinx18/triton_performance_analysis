import torch

def s118_pytorch(a, bb):
    """
    PyTorch implementation of TSVC s118 kernel.
    
    Original C code:
    for (int nl = 0; nl < 200*(iterations/LEN_2D); nl++) {
        for (int i = 1; i < LEN_2D; i++) {
            for (int j = 0; j <= i - 1; j++) {
                a[i] += bb[j][i] * a[i-j-1];
            }
        }
    }
    """
    a = a.contiguous()
    bb = bb.contiguous()
    
    LEN_2D = a.shape[0]
    
    for i in range(1, LEN_2D):
        for j in range(i):
            a[i] += bb[j, i] * a[i - j - 1]