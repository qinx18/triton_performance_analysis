import torch

def s341_pytorch(a, b):
    """
    PyTorch implementation of TSVC s341 - conditional vector compress/pack.
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        j = -1;
        for (int i = 0; i < LEN_1D; i++) {
            if (b[i] > (real_t)0.) {
                j++;
                a[j] = b[i];
            }
        }
    }
    
    Args:
        a: Output tensor (read-write)
        b: Input tensor (read-only)
    """
    a = a.contiguous()
    b = b.contiguous()
    
    # Find elements where b > 0
    mask = b > 0.0
    
    # Extract positive values from b
    positive_values = b[mask]
    
    # Pack the positive values into the beginning of array a
    num_positive = positive_values.numel()
    if num_positive > 0:
        a[:num_positive] = positive_values