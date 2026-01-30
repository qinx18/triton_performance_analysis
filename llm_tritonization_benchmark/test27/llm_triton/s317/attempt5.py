import triton
import triton.language as tl
import torch

@triton.jit
def s317_kernel(q_ptr, iterations: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        q = 1.0
        for i in range(iterations):
            q = q * 0.99
        tl.store(q_ptr, q)

def s317_triton(n):
    q = torch.zeros(1, dtype=torch.float32, device='cuda')
    iterations = n // 2
    
    s317_kernel[(1,)](q, iterations)
    
    return q[0].item()