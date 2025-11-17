import torch

def s352_pytorch(a, b):
    """
    PyTorch implementation of TSVC s352 - unrolled dot product.
    
    Original C code:
    for (int nl = 0; nl < 8*iterations; nl++) {
        dot = (real_t)0.;
        for (int i = 0; i < LEN_1D; i += 5) {
            dot = dot + a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2]
                * b[i + 2] + a[i + 3] * b[i + 3] + a[i + 4] * b[i + 4];
        }
    }
    
    Arrays used: a (r), b (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    
    dot = torch.tensor(0.0, dtype=a.dtype, device=a.device)
    
    # Handle the unrolled loop with step size 5
    len_1d = a.size(0)
    for i in range(0, len_1d, 5):
        # Calculate how many elements we can process in this iteration
        remaining = min(5, len_1d - i)
        
        if remaining >= 5:
            dot = dot + a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3] + a[i + 4] * b[i + 4]
        elif remaining == 4:
            dot = dot + a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3]
        elif remaining == 3:
            dot = dot + a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2]
        elif remaining == 2:
            dot = dot + a[i] * b[i] + a[i + 1] * b[i + 1]
        elif remaining == 1:
            dot = dot + a[i] * b[i]
    
    return a, b