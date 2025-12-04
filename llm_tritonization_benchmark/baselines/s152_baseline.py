import torch

def s152_pytorch(a, b, c, d, e):
    """
    PyTorch implementation of TSVC s152 kernel.

    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            b[i] = d[i] * e[i];
            s152s(a, b, c, i);  // s152s does: a[i] += b[i] * c[i]
        }
    }

    Helper function s152s:
    void s152s(real_t a[LEN_1D], real_t b[LEN_1D], real_t c[LEN_1D], int i) {
        a[i] += b[i] * c[i];
    }
    """
    # First compute b = d * e
    b[:] = d * e

    # Then a += b * c (this is what s152s does for each element)
    a[:] = a + b * c
