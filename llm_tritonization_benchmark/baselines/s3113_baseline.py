import torch

def s3113_pytorch(a):
    """
    PyTorch implementation of TSVC s3113 - finding maximum absolute value.

    Original C code:
    for (int nl = 0; nl < iterations*4; nl++) {
        max = ABS(a[0]);
        for (int i = 0; i < LEN_1D; i++) {
            if ((ABS(a[i])) > max) {
                max = ABS(a[i]);
            }
        }
    }

    Arrays: a (read only)
    """
    a = a.contiguous()

    # Find the maximum absolute value (pure reduction)
    max_val = torch.max(torch.abs(a))

    return max_val