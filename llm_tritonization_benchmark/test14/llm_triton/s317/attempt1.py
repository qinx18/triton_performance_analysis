import torch
import triton
import triton.language as tl

@triton.jit
def s317_kernel(q_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    if pid == 0:
        q = 1.0
        for i in range(n_elements):
            q *= 0.99
        tl.store(q_ptr, q)

def s317_triton():
    LEN_1D = 32000
    n_elements = LEN_1D // 2
    
    q_tensor = torch.zeros(1, dtype=torch.float32, device='cuda')
    
    grid = (1,)
    s317_kernel[grid](q_tensor, n_elements, BLOCK_SIZE=128)
    
    return q_tensor.item()