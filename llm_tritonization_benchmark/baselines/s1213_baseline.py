import torch

def s1213_pytorch(a, b, c, d):
    """
    Corrected PyTorch implementation of TSVC s1213

    Original C code:
    for (int i = 1; i < LEN_1D-1; i++) {
        a[i] = b[i-1] + c[i];
        b[i] = a[i+1] * d[i];
    }

    The issue: a[i] should use the UPDATED b[i-1], not the original.

    Solution: Regroup the loop iterations:
    - First:  a[1] = b[0] + c[1]  (b[0] never gets modified)
    - Middle: b[i] = a[i+1] * d[i]; a[i+1] = b[i] + c[i+1]
              (uses original a[i+1], then uses just-computed b[i])
    - Last:   b[n-2] = a[n-1] * d[n-2]  (a[n-1] never gets modified)

    The middle iterations can be fully parallelized because:
    - b[i] only depends on a[i+1] (original value)
    - a[i+1] uses the just-computed b[i] in the same iteration
    - Different indices mean no dependency between parallel threads
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()

    n = a.shape[0]

    if n < 3:
        return a, b

    # Save original a values for computing b
    a_orig = a.clone()

    # First iteration (special case): a[1] = b[0] + c[1]
    a[1] = b[0] + c[1]

    # Middle iterations (parallelizable): i from 1 to n-3
    # b[i] = a_orig[i+1] * d[i]
    # a[i+1] = b[i] + c[i+1]
    if n > 3:
        i = torch.arange(1, n - 2, device=a.device)
        # Compute b[i] using original a[i+1]
        b[i] = a_orig[i + 1] * d[i]
        # Compute a[i+1] using the just-computed b[i]
        a[i + 1] = b[i] + c[i + 1]

    # Last iteration (special case): b[n-2] = a[n-1] * d[n-2]
    b[n - 2] = a_orig[n - 1] * d[n - 2]

    return a, b
