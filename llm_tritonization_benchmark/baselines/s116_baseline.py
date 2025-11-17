import torch

def s116_pytorch(a):
    """
    PyTorch implementation of TSVC s116 - linear dependence testing, unrolling with no dependencies.
    
    Original C code:
    for (int nl = 0; nl < iterations*10; nl++) {
        for (int i = 0; i < LEN_1D - 5; i += 5) {
            a[i] = a[i + 1] * a[i];
            a[i + 1] = a[i + 2] * a[i + 1];
            a[i + 2] = a[i + 3] * a[i + 2];
            a[i + 3] = a[i + 4] * a[i + 3];
            a[i + 4] = a[i + 5] * a[i + 4];
        }
    }
    
    Args:
        a: Input/output tensor
        
    Returns:
        torch.Tensor: Modified array a
    """
    a = a.contiguous()
    n = a.size(0)
    
    # Process in chunks of 5
    for i in range(0, n - 5, 5):
        # Store intermediate values to handle dependencies correctly
        temp0 = a[i + 1] * a[i]
        temp1 = a[i + 2] * a[i + 1]
        temp2 = a[i + 3] * a[i + 2]
        temp3 = a[i + 4] * a[i + 3]
        temp4 = a[i + 5] * a[i + 4]
        
        # Update in order
        a[i] = temp0
        a[i + 1] = temp1
        a[i + 2] = temp2
        a[i + 3] = temp3
        a[i + 4] = temp4
    
    return a