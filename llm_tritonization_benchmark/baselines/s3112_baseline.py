import torch

def s3112_pytorch(a, b):
    """
    PyTorch implementation of TSVC s3112 - cumulative sum operation.
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        sum = (real_t)0.0;
        for (int i = 0; i < LEN_1D; i++) {
            sum += a[i];
            b[i] = sum;
        }
    }
    
    Arrays: a (r), b (rw)
    """
    a = a.contiguous()
    b = b.contiguous()
    
    # Compute cumulative sum
    b = torch.cumsum(a, dim=0)
    
    return b