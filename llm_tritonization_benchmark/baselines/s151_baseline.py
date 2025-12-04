import torch

def s151_pytorch(a, b):
    """
    PyTorch implementation of TSVC s151.

    Original C code:
    for (int nl = 0; nl < 5*iterations; nl++) {
        s151s(a, b, 1);
    }

    Helper function s151s:
    void s151s(real_t a[LEN_1D], real_t b[LEN_1D], int m) {
        for (int i = 0; i < LEN_1D-1; i++) {
            a[i] = a[i + m] + b[i];
        }
    }

    With m=1: a[i] = a[i+1] + b[i] for i in 0..LEN_1D-2
    """
    n = a.shape[0]
    # a[i] = a[i+1] + b[i] for i in 0..n-2
    # Since we read from i+1 and write to i, no dependency within loop
    a[:-1] = a[1:].clone() + b[:-1]
