import torch

def s212_pytorch(a, b, c, d):
    """
    Corrected PyTorch implementation of TSVC s212

    Original C code:
    for (int i = 0; i < LEN_1D-1; i++) {
        a[i] *= c[i];          // Modifies a[i]
        b[i] += a[i + 1] * d[i];  // Reads a[i+1] (original value!)
    }

    The key is that when we read a[i+1], we need the ORIGINAL value,
    not a value that's already been modified by a previous iteration.

    Must save original `a` before modifying it, just like s211 does with `b`.
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()

    # Save original a values before modification
    a_orig = a.clone()

    # First update a[i] *= c[i] for i in range(len-1)
    a[:-1] *= c[:-1]

    # Then update b[i] += a_orig[i + 1] * d[i] (use ORIGINAL a values)
    b[:-1] += a_orig[1:] * d[:-1]

    return a, b
