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
    
    # Perform the computation 4 times as in the original loop (iterations*4)
    for _ in range(4):
        max_val = torch.abs(a[0])
        for i in range(a.size(0)):
            abs_val = torch.abs(a[i])
            if abs_val > max_val:
                max_val = abs_val