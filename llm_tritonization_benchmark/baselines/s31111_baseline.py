import torch

def s31111_pytorch(a, test):
    """
    PyTorch implementation of TSVC s31111.
    
    Original C code:
    for (int nl = 0; nl < 2000*iterations; nl++) {
        sum = (real_t)0.;
        sum += test(a);
        sum += test(&a[4]);
        sum += test(&a[8]);
        sum += test(&a[12]);
        sum += test(&a[16]);
        sum += test(&a[20]);
        sum += test(&a[24]);
        sum += test(&a[28]);
    }
    
    Args:
        a: Input tensor (read-only)
        test: Function to apply to tensor segments
    """
    a = a.contiguous()
    
    for _ in range(2000):
        sum_val = 0.0
        sum_val += test(a)
        sum_val += test(a[4:])
        sum_val += test(a[8:])
        sum_val += test(a[12:])
        sum_val += test(a[16:])
        sum_val += test(a[20:])
        sum_val += test(a[24:])
        sum_val += test(a[28:])