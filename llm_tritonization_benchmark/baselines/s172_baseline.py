import torch

def s172_pytorch(a, b, n1, n3):
    """
    PyTorch implementation of TSVC s172.
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = n1-1; i < LEN_1D; i += n3) {
            a[i] += b[i];
        }
    }
    
    Arrays: a (rw), b (r)
    Scalars: n1, n3
    """
    a = a.contiguous()
    b = b.contiguous()
    
    # Create indices for the strided access pattern
    start_idx = n1 - 1
    indices = torch.arange(start_idx, a.size(0), n3, dtype=torch.long, device=a.device)
    
    # Update a at the strided indices
    if len(indices) > 0:
        a[indices] += b[indices]
    
    return a