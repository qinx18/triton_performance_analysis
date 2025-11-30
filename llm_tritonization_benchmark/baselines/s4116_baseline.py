import torch

def s4116_pytorch(a, aa, ip, inc, j):
    """
    PyTorch implementation of TSVC s4116
    
    Original C code:
    for (int nl = 0; nl < 100*iterations; nl++) {
        sum = 0.;
        for (int i = 0; i < LEN_2D-1; i++) {
            off = inc + i;
            sum += a[off] * aa[j-1][ip[i]];
        }
    }
    
    Arrays: a (r), aa (r), ip (r)
    Scalars: inc, j
    """
    a = a.contiguous()
    aa = aa.contiguous()
    ip = ip.contiguous()
    
    LEN_2D = aa.shape[1]
    
    sum_val = 0.0
    for i in range(LEN_2D - 1):
        off = inc + i
        sum_val += a[off] * aa[j-1][ip[i]]