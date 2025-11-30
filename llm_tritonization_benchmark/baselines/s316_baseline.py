import torch

def s316_pytorch(a):
    """
    TSVC s316 - Find minimum element in array
    
    Original C code:
    for (int nl = 0; nl < iterations*5; nl++) {
        x = a[0];
        for (int i = 1; i < LEN_1D; ++i) {
            if (a[i] < x) {
                x = a[i];
            }
        }
    }
    
    Arrays: a (read-only)
    """
    a = a.contiguous()
    
    # Perform the computation 5 times as in original (iterations*5 with iterations=1)
    for _ in range(5):
        x = a[0]
        for i in range(1, len(a)):
            if a[i] < x:
                x = a[i]