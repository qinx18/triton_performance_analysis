import torch
import triton
import triton.language as tl

@triton.jit
def s4115_kernel(a_ptr, b_ptr, ip_ptr, partial_sums_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load direct array and indices
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    indices = tl.load(ip_ptr + offsets, mask=mask, other=0)
    
    # GATHER: Load from scattered locations using indices
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    
    # Parallel multiply and reduce
    block_sum = tl.sum(a_vals * b_vals, axis=0)
    tl.store(partial_sums_ptr + pid, block_sum)

def s4115_triton(a, b, ip):
    n = a.shape[0]
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    partial_sums = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    s4115_kernel[(num_blocks,)](a, b, ip, partial_sums, n, BLOCK_SIZE)
    return partial_sums.sum().item()