import torch

def s1244_pytorch(a, b, c, d):
    """
    PyTorch implementation of TSVC s1244

    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D-1; i++) {
            a[i] = b[i] + c[i] * c[i] + b[i]*b[i] + c[i];
            d[i] = a[i] + a[i+1];
        }
    }

    Critical dependency: d[i] uses NEW a[i] and OLD a[i+1]
    (a[i+1] hasn't been updated yet in iteration i)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()

    # Process sequentially to respect the dependency
    for i in range(len(a) - 1):
        a[i] = b[i] + c[i] * c[i] + b[i] * b[i] + c[i]
        d[i] = a[i] + a[i+1]  # NEW a[i] + OLD a[i+1]

    return a, d