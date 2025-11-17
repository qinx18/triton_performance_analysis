import torch

def s453_pytorch(a, b):
    """
    PyTorch implementation of TSVC s453.
    
    Original C code:
    for (int nl = 0; nl < iterations*2; nl++) {
        s = 0.;
        for (int i = 0; i < LEN_1D; i++) {
            s += (real_t)2.;
            a[i] = s * b[i];
        }
    }
    
    Arrays: a (rw), b (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    
    len_1d = a.size(0)
    
    # Create cumulative sum of 2's: s = 2, 4, 6, 8, ...
    s_values = torch.arange(1, len_1d + 1, dtype=a.dtype, device=a.device) * 2.0
    
    # Apply the computation: a[i] = s * b[i]
    a = s_values * b
    
    return a