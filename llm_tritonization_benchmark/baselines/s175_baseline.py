import torch

def s175_pytorch(a, b, inc):
    """
    PyTorch implementation of TSVC s175.
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D-1; i += inc) {
            a[i] = a[i + inc] + b[i];
        }
    }
    
    Arrays: a (rw), b (r)
    Scalar parameters: inc
    """
    a = a.contiguous()
    b = b.contiguous()
    
    len_1d = a.size(0)
    
    # Generate indices for the strided loop
    indices = torch.arange(0, len_1d - 1, inc, device=a.device, dtype=torch.long)
    
    # Vectorized computation: a[i] = a[i + inc] + b[i]
    a[indices] = a[indices + inc] + b[indices]
    
    return a