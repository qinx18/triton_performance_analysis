import torch

def s1113_pytorch(a, b):
    """
    PyTorch implementation of TSVC s1113

    Original C code:
    for (int i = 0; i < LEN_1D; i++) {
        a[i] = a[LEN_1D/2] + b[i];
    }

    IMPORTANT: a[LEN_1D/2] is updated at iteration i=LEN_1D/2, and subsequent
    iterations use the UPDATED value.

    Sequential execution:
    - i = 0 to mid-1: a[i] = orig_a[mid] + b[i]
    - i = mid: a[mid] = orig_a[mid] + b[mid]  (a[mid] UPDATED!)
    - i = mid+1 to end: a[i] = new_a[mid] + b[i] = (orig_a[mid] + b[mid]) + b[i]
    """
    a = a.contiguous()
    b = b.contiguous()

    LEN_1D = a.size(0)
    mid = LEN_1D // 2

    # Save original a[mid] before it gets updated
    orig_a_mid = a[mid].clone()

    # i = 0 to mid (inclusive): uses original a[mid]
    a[:mid+1] = orig_a_mid + b[:mid+1]

    # i = mid+1 to end: uses UPDATED a[mid] (which is now orig_a_mid + b[mid])
    a[mid+1:] = a[mid] + b[mid+1:]