import triton
import triton.language as tl
import torch

@triton.jit
def s342_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load a values
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask)
    
    # Count positive elements before this block
    j_offset = 0
    for block_idx in range(pid):
        prev_block_start = block_idx * BLOCK_SIZE
        prev_offsets = prev_block_start + offsets
        prev_mask = prev_offsets < n_elements
        prev_a_vals = tl.load(a_ptr + prev_offsets, mask=prev_mask)
        positive_count = tl.sum((prev_a_vals > 0.0).to(tl.int32))
        j_offset += positive_count
    
    # Process current block
    for i in range(BLOCK_SIZE):
        if block_start + i < n_elements:
            a_val = tl.load(a_ptr + block_start + i)
            if a_val > 0.0:
                b_val = tl.load(b_ptr + j_offset)
                tl.store(a_ptr + block_start + i, b_val)
                j_offset += 1

def s342_triton(a, b):
    n_elements = a.numel()
    BLOCK_SIZE = 128
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s342_kernel[grid](a, b, n_elements, BLOCK_SIZE)