import torch

def s352_pytorch(a, b):
    """
    PyTorch implementation of TSVC s352 - dot product with loop unrolling by 5
    
    Original C code:
    for (int nl = 0; nl < 8*iterations; nl++) {
        dot = (real_t)0.;
        for (int i = 0; i < LEN_1D; i += 5) {
            dot = dot + a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2]
                * b[i + 2] + a[i + 3] * b[i + 3] + a[i + 4] * b[i + 4];
        }
    }
    
    Arrays used: a (r), b (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    
    # Ensure array length is divisible by 5 for proper unrolling
    len_1d = a.shape[0]
    len_unroll = (len_1d // 5) * 5
    
    if len_unroll > 0:
        # Reshape arrays to handle unrolling by 5
        a_unroll = a[:len_unroll].view(-1, 5)
        b_unroll = b[:len_unroll].view(-1, 5)
        
        # Compute dot product with unrolling
        dot = torch.sum(a_unroll * b_unroll)
    else:
        dot = torch.tensor(0.0, dtype=a.dtype, device=a.device)
    
    return dot