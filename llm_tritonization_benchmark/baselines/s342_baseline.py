import torch

def s342_pytorch(a, b):
    """
    PyTorch implementation of TSVC s342 - search loop with conditional assignment.
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        j = -1;
        for (int i = 0; i < LEN_1D; i++) {
            if (a[i] > (real_t)0.) {
                j++;
                a[i] = b[j];
            }
        }
    }
    
    Args:
        a: Input/output tensor (read-write)
        b: Input tensor (read)
    """
    a = a.contiguous()
    b = b.contiguous()
    
    # Find indices where a[i] > 0
    positive_mask = a > 0.0
    positive_indices = torch.where(positive_mask)[0]
    
    # For each positive element, assign corresponding b value
    for idx, pos_idx in enumerate(positive_indices):
        a[pos_idx] = b[idx]