import torch

def s3111_pytorch(a):
    """
    PyTorch implementation of TSVC s3111 - conditional sum reduction.
    
    Original C code:
    for (int nl = 0; nl < iterations/2; nl++) {
        sum = 0.;
        for (int i = 0; i < LEN_1D; i++) {
            if (a[i] > (real_t)0.) {
                sum += a[i];
            }
        }
    }
    """
    a = a.contiguous()
    
    # Conditional sum: sum all positive elements
    positive_mask = a > 0.0
    sum_val = torch.sum(torch.where(positive_mask, a, torch.tensor(0.0, device=a.device, dtype=a.dtype)))
    
    return sum_val