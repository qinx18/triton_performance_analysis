import torch

def s4115_pytorch(a, b, ip):
    """
    PyTorch implementation of TSVC s4115 kernel.
    
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
    
    # Ensure ip indices are within bounds and convert to long for indexing
    ip_indices = ip.long()
    
    # Compute sum of a[i] * b[ip[i]] for all i
    sum_val = torch.sum(a * b[ip_indices])
    
    return sum_val