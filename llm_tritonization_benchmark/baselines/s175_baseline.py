import torch

def s175_pytorch(a, b, inc):
    """
    PyTorch implementation of TSVC s175 kernel.
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D-1; i += inc) {
            a[i] = a[i + inc] + b[i];
        }
    }
    
    Arrays: a (rw), b (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    
    len_1d = a.shape[0]
    indices = torch.arange(0, len_1d - 1, inc, device=a.device)
    
    # Only update indices that don't go out of bounds when accessing a[i + inc]
    valid_indices = indices[indices + inc < len_1d]
    
    a[valid_indices] = a[valid_indices + inc] + b[valid_indices]