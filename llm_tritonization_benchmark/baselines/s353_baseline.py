import torch

def s353_pytorch(a, b, ip, alpha):
    """
    PyTorch implementation of TSVC s353 - unrolled saxpy with indirect addressing.
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i += 5) {
            a[i] += alpha * b[ip[i]];
            a[i + 1] += alpha * b[ip[i + 1]];
            a[i + 2] += alpha * b[ip[i + 2]];
            a[i + 3] += alpha * b[ip[i + 3]];
            a[i + 4] += alpha * b[ip[i + 4]];
        }
    }
    
    Arrays: a (rw), b (r), ip (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    ip = ip.contiguous()
    
    LEN_1D = a.shape[0]
    
    for i in range(0, LEN_1D, 5):
        if i < LEN_1D:
            a[i] += alpha * b[ip[i]]
        if i + 1 < LEN_1D:
            a[i + 1] += alpha * b[ip[i + 1]]
        if i + 2 < LEN_1D:
            a[i + 2] += alpha * b[ip[i + 2]]
        if i + 3 < LEN_1D:
            a[i + 3] += alpha * b[ip[i + 3]]
        if i + 4 < LEN_1D:
            a[i + 4] += alpha * b[ip[i + 4]]
    
    return a