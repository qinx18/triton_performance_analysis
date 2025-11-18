import torch

def s221_pytorch(a, b, c, d):
    """
    Optimized PyTorch implementation of TSVC s221

    Original C code:
    for (int i = 1; i < LEN_1D; i++) {
        a[i] += c[i] * d[i];           // Parallelizable
        b[i] = b[i-1] + a[i] + d[i];   // Cumulative sum pattern
    }

    Key insight: Split into two independent steps:
    1. Update all a[i] in parallel (no dependencies)
    2. Compute b[i] using cumulative sum (prefix sum)

    The b computation is:
    b[i] = b[i-1] + a[i] + d[i]
    b[i] = b[0] + sum(a[1:i+1] + d[1:i+1])

    This is a cumulative sum that can be computed efficiently!
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()

    n = a.shape[0]

    if n < 2:
        return a, b

    # Step 1: Update all a[i] in parallel (fully parallelizable)
    a[1:] += c[1:] * d[1:]

    # Step 2: Compute b[i] using cumulative sum
    # b[i] = b[i-1] + a[i] + d[i]
    # This is equivalent to:
    # b[1:] = b[0] + cumsum(a[1:] + d[1:])

    increments = a[1:] + d[1:]
    b[1:] = b[0] + torch.cumsum(increments, dim=0)

    return a, b
