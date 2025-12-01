import torch

def s161_pytorch(a, b, c, d, e):
    """
    Corrected PyTorch implementation of TSVC s161 - following true C sequential semantics.

    Original C code:
    for (int i = 0; i < LEN_1D-1; ++i) {
        if (b[i] < (real_t)0.) {
            goto L20;
        }
        a[i] = c[i] + d[i] * e[i];
        goto L10;
L20:
        c[i+1] = a[i] + d[i] * d[i];
L10:
        ;
    }

    The correct implementation must:
    1. First process all b[i] < 0 (write c[i+1])
    2. Then process all b[i] >= 0 (read potentially updated c[i])
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()

    n = len(a) - 1

    # Create masks for the conditional
    mask_neg = b[:n] < 0.0
    mask_pos = ~mask_neg

    # Phase 1: When b[i] < 0, write c[i+1] = a[i] + d[i] * d[i]
    # This must happen FIRST so that phase 2 can read updated c values
    c[1:n+1] = torch.where(mask_neg, a[:n] + d[:n] * d[:n], c[1:n+1])

    # Phase 2: When b[i] >= 0, write a[i] = c[i] + d[i] * e[i]
    # This reads c[i] which may have been updated in phase 1
    a[:n] = torch.where(mask_pos, c[:n] + d[:n] * e[:n], a[:n])

    return a, c
