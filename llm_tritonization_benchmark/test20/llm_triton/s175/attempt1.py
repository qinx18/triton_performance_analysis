import triton
import triton.language as tl
import torch

@triton.jit
def s175_kernel(a_ptr, a_copy_ptr, b_ptr, inc, n, BLOCK_SIZE: tl.constexpr):
    # Calculate which iterations this block will handle
    block_id = tl.program_id(0)
    
    # Each block processes a range of iterations
    iterations_per_block = triton.cdiv(BLOCK_SIZE, inc)
    start_iter = block_id * iterations_per_block
    
    # Calculate the actual i values for this block
    iter_offsets = tl.arange(0, iterations_per_block)
    i_values = start_iter * inc + iter_offsets * inc
    
    # Mask for valid iterations
    valid_mask = i_values < (n - 1)
    
    # Load values
    b_vals = tl.load(b_ptr + i_values, mask=valid_mask)
    a_read_vals = tl.load(a_copy_ptr + i_values + inc, mask=valid_mask)
    
    # Compute result
    result = a_read_vals + b_vals
    
    # Store result
    tl.store(a_ptr + i_values, result, mask=valid_mask)

def s175_triton(a, b, inc):
    n = a.shape[0]
    
    # Create read-only copy for WAR race condition handling
    a_copy = a.clone()
    
    # Calculate grid size
    BLOCK_SIZE = 256
    max_iterations = triton.cdiv(n - 1, inc)
    iterations_per_block = triton.cdiv(BLOCK_SIZE, inc)
    grid = (triton.cdiv(max_iterations, iterations_per_block),)
    
    # Launch kernel
    s175_kernel[grid](
        a, a_copy, b, inc, n,
        BLOCK_SIZE=BLOCK_SIZE
    )