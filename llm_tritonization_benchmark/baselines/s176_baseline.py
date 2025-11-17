import torch

def s176_pytorch(a, b, c, iterations, m):
    """
    PyTorch implementation of TSVC s176
    
    Original C code:
    for (int nl = 0; nl < 4*(iterations/LEN_1D); nl++) {
        for (int j = 0; j < (LEN_1D/2); j++) {
            for (int i = 0; i < m; i++) {
                a[i] += b[i+m-j-1] * c[j];
            }
        }
    }
    
    Arrays: a (rw), b (r), c (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    
    LEN_1D = len(c)
    
    for j in range(LEN_1D // 2):
        for i in range(m):
            a[i] += b[i + m - j - 1] * c[j]
    
    return a