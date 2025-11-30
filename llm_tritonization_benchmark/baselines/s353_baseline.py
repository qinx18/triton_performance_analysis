import torch

def s353_pytorch(a, b, ip, alpha):
    """
    PyTorch implementation of TSVC s353 - vectorized indirect addressing with unrolling.
    
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
    
    Args:
        a: read-write tensor
        b: read-only tensor
        ip: read-only index tensor
        alpha: scalar parameter
    """
    a = a.contiguous()
    b = b.contiguous()
    ip = ip.contiguous()
    
    n = a.size(0)
    
    # Process elements in groups of 5
    for i in range(0, n - 4, 5):
        a[i] += alpha * b[ip[i]]
        a[i + 1] += alpha * b[ip[i + 1]]
        a[i + 2] += alpha * b[ip[i + 2]]
        a[i + 3] += alpha * b[ip[i + 3]]
        a[i + 4] += alpha * b[ip[i + 4]]
    
    # Handle remaining elements
    remaining = n % 5
    if remaining > 0:
        start_idx = n - remaining
        for i in range(start_idx, n):
            a[i] += alpha * b[ip[i]]