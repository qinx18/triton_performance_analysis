import torch

def vas_pytorch(a, b, ip):
    """
    PyTorch implementation of TSVC vas function.

    Original C code:
    for (int nl = 0; nl < 2*iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[ip[i]] = b[i];
        }
    }

    IMPORTANT: Must process sequentially to match C semantics.
    When there are duplicate indices in ip, last write wins.
    PyTorch's a[ip] = b has undefined behavior for duplicates.

    Arrays: a (rw), b (r), ip (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    ip = ip.contiguous()

    # Process sequentially to ensure "last write wins" for duplicates
    for i in range(len(ip)):
        a[ip[i]] = b[i]

    return a
