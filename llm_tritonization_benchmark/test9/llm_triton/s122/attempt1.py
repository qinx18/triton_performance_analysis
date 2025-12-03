import triton
import triton.language as tl
import torch

@triton.jit
def s122_kernel(a_ptr, b_ptr, n1, n3, LEN_1D, BLOCK_SIZE: tl.constexpr):
    # Calculate number of iterations in the loop
    num_iterations = (LEN_1D - (n1 - 1) + n3 - 1) // n3
    
    # Get block start
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    # Check if this block has work to do
    if block_start >= num_iterations:
        return
    
    # Create offset vector once
    offsets = tl.arange(0, BLOCK_SIZE)
    block_offsets = block_start + offsets
    mask = block_offsets < num_iterations
    
    # For each valid iteration in this block
    for idx in range(BLOCK_SIZE):
        if block_start + idx >= num_iterations:
            break
            
        iteration_idx = block_start + idx
        
        # Calculate i value for this iteration
        i = (n1 - 1) + iteration_idx * n3
        
        # Calculate k value: k starts at 0, increments by j=1 each iteration
        k = iteration_idx + 1  # j=1, so k = 0 + 1*1 + 1*1 + ... = iteration_idx + 1
        
        # Load from a[i]
        a_val = tl.load(a_ptr + i)
        
        # Load from b[LEN_1D - k]
        b_idx = LEN_1D - k
        b_val = tl.load(b_ptr + b_idx)
        
        # Update a[i]
        new_val = a_val + b_val
        tl.store(a_ptr + i, new_val)

def s122_triton(a, b, n1, n3):
    LEN_1D = a.shape[0]
    
    # Calculate number of iterations in the loop
    num_iterations = (LEN_1D - (n1 - 1) + n3 - 1) // n3
    
    if num_iterations <= 0:
        return
    
    BLOCK_SIZE = 128
    grid = (triton.cdiv(num_iterations, BLOCK_SIZE),)
    
    s122_kernel[grid](
        a, b, n1, n3, LEN_1D, BLOCK_SIZE
    )