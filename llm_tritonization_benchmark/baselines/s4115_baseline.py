import torch

def s4115_pytorch(a, b, ip):
    """
    PyTorch implementation of TSVC s4115 - indirect addressing with dot product.
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        sum = 0.;
        for (int i = 0; i < LEN_1D; i++) {
            sum += a[i] * b[ip[i]];
        }
    }
    
    Arrays: a (r), b (r), ip (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    ip = ip.contiguous()
    
    # Compute sum using indirect addressing
    sum_val = torch.sum(a * b[ip])
    
    return a, b, ip