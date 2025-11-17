import torch

def s482_pytorch(a, b, c):
    """
    PyTorch implementation of TSVC s482 function.
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[i] += b[i] * c[i];
            if (c[i] > b[i]) break;
        }
    }
    
    Arrays used: a (rw), b (r), c (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    
    # Find the first index where c[i] > b[i]
    condition = c > b
    if torch.any(condition):
        break_idx = torch.argmax(condition.int())
        # Process elements up to (but not including) the break point
        a[:break_idx] += b[:break_idx] * c[:break_idx]
    else:
        # No break condition met, process all elements
        a += b * c
    
    return a