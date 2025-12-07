import torch
import triton
import triton.language as tl

@triton.jit
def s3113_kernel(a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    block_max_abs = float('-inf')
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        abs_vals = tl.abs(vals)
        current_max = tl.max(abs_vals)
        block_max_abs = tl.maximum(block_max_abs, current_max)
    
    tl.store(tl.program_id(0) + tl.zeros([1], dtype=tl.int32), block_max_abs)

def s3113_triton(a):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    
    # Single program version since we need global max
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    s3113_kernel[(1,)](
        a, n_elements, output, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output.item()