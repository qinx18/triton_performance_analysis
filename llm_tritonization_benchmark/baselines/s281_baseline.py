import torch

def s281_pytorch(a, b, c):
    """
    CORRECT PyTorch implementation of TSVC s281

    Original C code:
    for (int i = 0; i < LEN_1D; i++) {
        x = a[LEN_1D-i-1] + b[i] * c[i];
        a[i] = x - 1.0;
        b[i] = x;
    }

    Key dependency: At i >= LEN_1D/2, we read a[LEN_1D-i-1] which was WRITTEN earlier!
    - i=0: Read a[LEN_1D-1] (original), Write a[0]
    - ...
    - i=LEN_1D/2: Read a[LEN_1D/2-1] (UPDATED from iteration i=LEN_1D/2-1), Write a[LEN_1D/2]

    Must process sequentially to get correct behavior!
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()

    LEN_1D = a.shape[0]

    # Must process sequentially due to read-after-write dependency
    for i in range(LEN_1D):
        x = a[LEN_1D - i - 1] + b[i] * c[i]
        a[i] = x - 1.0
        b[i] = x

    return a, b
