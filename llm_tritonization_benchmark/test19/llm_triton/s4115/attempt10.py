import triton
import triton.language as tl
import torch

@triton.jit
def s4115_kernel(a_ptr, b_ptr, ip_ptr, output_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < N
    
    a_vals = tl.load(a_ptr + indices, mask=mask, other=0.0)
    ip_vals = tl.load(ip_ptr + indices, mask=mask, other=0)
    
    ip_mask = mask & (ip_vals >= 0) & (ip_vals < N)
    b_vals = tl.load(b_ptr + ip_vals, mask=ip_mask, other=0.0)
    
    products = tl.where(mask, a_vals * b_vals, 0.0)
    
    tl.store(output_ptr + indices, products, mask=mask)

def s4115_triton(a, b, ip):
    N = a.shape[0]
    
    output = torch.zeros_like(a)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s4115_kernel[grid](
        a, b, ip, output,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return float(output.sum().item())