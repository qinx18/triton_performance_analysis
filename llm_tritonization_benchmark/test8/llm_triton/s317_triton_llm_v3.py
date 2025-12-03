import torch
import triton
import triton.language as tl

@triton.jit
def s317_kernel(output_ptr):
    q = 1.0
    for i in range(16000):  # LEN_1D/2 where LEN_1D is typically 32000
        q *= 0.99
    tl.store(output_ptr, q)

def s317_triton():
    output = torch.zeros(1, dtype=torch.float32, device='cuda')
    grid = (1,)
    s317_kernel[grid](output)
    return output.item()