import triton
import triton.language as tl
import torch

@triton.jit
def s292_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Since this has sequential dependencies, process all elements in sequence
    # Each block handles a portion but maintains the dependency pattern
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE) + block_start
    mask = offsets < n_elements
    
    # For each element in this block, we need to compute based on the dependency pattern
    for local_idx in range(BLOCK_SIZE):
        i = block_start + local_idx
        if i >= n_elements:
            return
            
        # Calculate im1 and im2 based on the pattern from C code
        # At start: im1 = n-1, im2 = n-2
        # After each iteration: im2 = old_im1, im1 = i
        if i == 0:
            im1 = n_elements - 1
            im2 = n_elements - 2
        elif i == 1:
            im1 = 0  # previous i
            im2 = n_elements - 1  # previous im1
        else:
            im1 = i - 1
            im2 = i - 2
        
        # Load values
        b_i = tl.load(b_ptr + i)
        b_im1 = tl.load(b_ptr + im1)
        b_im2 = tl.load(b_ptr + im2)
        
        # Compute result
        result = (b_i + b_im1 + b_im2) * 0.333
        
        # Store result
        tl.store(a_ptr + i, result)

def s292_triton(a, b):
    n_elements = b.shape[0]
    BLOCK_SIZE = 256
    
    # Use single thread block to maintain sequential dependencies
    grid = (1,)
    s292_kernel[grid](a, b, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return a