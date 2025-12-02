import torch
import triton
import triton.language as tl

@triton.jit
def s122_kernel(a_ptr, b_ptr, n1, n3, len_1d, BLOCK_SIZE: tl.constexpr):
    # Calculate the number of iterations sequentially
    num_iters = 0
    i = n1 - 1
    while i < len_1d:
        num_iters += 1
        i += n3
    
    # Sequential execution with blocks
    block_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, num_iters, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < num_iters
        
        # Calculate i and k for each iteration
        j = 1
        k_vals = j * (current_offsets + 1)  # k = j * (iteration + 1)
        i_vals = (n1 - 1) + n3 * current_offsets
        
        # Load a values
        a_offsets = i_vals
        a_mask = mask & (a_offsets >= 0) & (a_offsets < len_1d)
        a_vals = tl.load(a_ptr + a_offsets, mask=a_mask, other=0.0)
        
        # Load b values
        b_offsets = len_1d - k_vals
        b_mask = mask & (b_offsets >= 0) & (b_offsets < len_1d)
        b_vals = tl.load(b_ptr + b_offsets, mask=b_mask, other=0.0)
        
        # Compute and store result
        result = a_vals + b_vals
        tl.store(a_ptr + a_offsets, result, mask=a_mask)

def s122_triton(a, b, n1, n3):
    len_1d = a.shape[0]
    
    # Calculate number of iterations
    num_iters = 0
    i = n1 - 1
    while i < len_1d:
        num_iters += 1
        i += n3
    
    if num_iters == 0:
        return
    
    BLOCK_SIZE = 256
    grid = (1,)  # Sequential execution
    
    s122_kernel[grid](
        a, b, n1, n3, len_1d, BLOCK_SIZE
    )