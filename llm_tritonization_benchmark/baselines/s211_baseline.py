import torch

def s211_pytorch(a, b, c, d, e):
    """
    Corrected PyTorch implementation of TSVC s211

    Original C code:
    for (int i = 1; i < LEN_1D-1; i++) {
        a[i] = b[i-1] + c[i] * d[i];
        b[i] = b[i+1] - e[i] * d[i];
    }

    The issue: a[i] should use the UPDATED b[i-1], not the original.

    Solution: Regroup the loop iterations:
    - First:  a[1] = b[0] + c[1]*d[1]  (b[0] never gets modified)
    - Middle: b[i] = b[i+1] - e[i]*d[i]; a[i+1] = b[i] + c[i+1]*d[i+1]
    - Last:   b[n-2] = b[n-1] - e[n-2]*d[n-2]

    The middle iterations can be fully parallelized because:
    - b[i] only depends on b[i+1] (original value)
    - a[i+1] uses the just-computed b[i] in the same iteration
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()

    n = a.shape[0]

    if n < 3:
        return a, b

    # Save original b[i+1] values for computing b[i]
    b_orig = b.clone()

    # First iteration (special case): a[1] = b[0] + c[1]*d[1]
    a[1] = b[0] + c[1] * d[1]

    # Middle iterations (parallelizable): i from 1 to n-3
    # b[i] = b[i+1] - e[i]*d[i]
    # a[i+1] = b[i] + c[i+1]*d[i+1]
    if n > 3:
        i = torch.arange(1, n - 2, device=a.device)
        # Compute b[i] using original b[i+1]
        b[i] = b_orig[i + 1] - e[i] * d[i]
        # Compute a[i+1] using the just-updated b[i]
        a[i + 1] = b[i] + c[i + 1] * d[i + 1]

    # Last iteration (special case): b[n-2] = b[n-1] - e[n-2]*d[n-2]
    b[n - 2] = b_orig[n - 1] - e[n - 2] * d[n - 2]

    return a, b
