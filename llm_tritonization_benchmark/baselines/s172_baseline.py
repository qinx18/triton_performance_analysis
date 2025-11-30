import torch

def s172_pytorch(a, b, n1, n3):
    """
    PyTorch implementation of TSVC s172
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = n1-1; i < LEN_1D; i += n3) {
            a[i] += b[i];
        }
    }
    
    Arrays: a (rw), b (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    
    # Generate indices: start from n1-1, increment by n3, until end of array
    indices = torch.arange(n1-1, a.size(0), n3, device=a.device)
    
    # Perform the computation: a[i] += b[i]
    a[indices] += b[indices]