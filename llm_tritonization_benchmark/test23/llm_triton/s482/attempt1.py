import torch
import triton
import triton.language as tl

@triton.jit
def s482_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    # Check if any element in this block violates the condition
    exit_condition = c_vals > b_vals
    
    # Only perform computation if we haven't hit exit condition
    # In SIMD execution, we need to mask out elements after first exit
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    result = a_vals + b_vals * c_vals
    tl.store(a_ptr + offsets, result, mask=mask)

def s482_triton(a, b, c):
    N = a.shape[0]
    
    # First, find the exit point on CPU
    exit_mask = c > b
    if torch.any(exit_mask):
        exit_idx = torch.argmax(exit_mask.int()).item()
        # Process only up to and including the exit index
        valid_N = exit_idx + 1
    else:
        # No exit condition, process all elements
        valid_N = N
    
    if valid_N > 0:
        BLOCK_SIZE = 256
        grid = (triton.cdiv(valid_N, BLOCK_SIZE),)
        s482_kernel[grid](a, b, c, valid_N, BLOCK_SIZE=BLOCK_SIZE)