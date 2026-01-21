import triton
import triton.language as tl
import torch

@triton.jit
def s313_kernel(a_ptr, b_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask, other=0.0)
    
    block_dot = tl.sum(a_vals * b_vals, axis=0)
    tl.store(output_ptr + tl.program_id(0), block_dot)

def s313_triton(a, b):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    n_blocks = triton.cdiv(N, BLOCK_SIZE)
    partial_sums = torch.zeros(n_blocks, dtype=a.dtype, device=a.device)
    
    grid = (n_blocks,)
    s313_kernel[grid](a, b, partial_sums, N, BLOCK_SIZE=BLOCK_SIZE)
    
    dot = torch.sum(partial_sums).item()
    return dot