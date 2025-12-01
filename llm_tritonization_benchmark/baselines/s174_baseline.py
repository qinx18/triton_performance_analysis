import torch

def s174_pytorch(a, b, M):
    """
    PyTorch implementation of TSVC s174

    Original C code:
    int M = *(int*)func_args->arg_info;
    for (int nl = 0; nl < 10*iterations; nl++) {
        for (int i = 0; i < M; i++) {
            a[i+M] = a[i] + b[i];
        }
    }

    Arrays: a (rw), b (r)
    Scalar: M (determines iteration count and offset)
    Note: Array a must be at least 2*M in size
    """
    a = a.contiguous()
    b = b.contiguous()

    a[M:2*M] = a[:M] + b[:M]