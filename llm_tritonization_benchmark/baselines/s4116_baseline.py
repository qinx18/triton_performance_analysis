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

    # Vectorized implementation
    offsets = inc + torch.arange(LEN_2D - 1, device=a.device)
    indices = ip[:LEN_2D - 1].long()
    sum_val = torch.sum(a[offsets] * aa[j - 1][indices])

    return sum_val