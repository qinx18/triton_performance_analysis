import torch
import triton
import triton.language as tl

@triton.jit
def s122_kernel(a_ptr, b_ptr, n1, n3, LEN_1D, BLOCK_SIZE: tl.constexpr):
    # Calculate how many iterations the original loop would have
    num_iterations = (LEN_1D - (n1 - 1) + n3 - 1) // n3
    
    # Get block ID and offsets
    block_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process iterations in blocks
    for block_start in range(0, num_iterations, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < num_iterations
        
        # Calculate original i values: i = n1-1 + iteration*n3
        i_vals = (n1 - 1) + current_offsets * n3
        i_mask = mask & (i_vals < LEN_1D)
        
        # Calculate k values: k = 1 + 2 + ... + (iteration+1) = (iteration+1)*(iteration+2)/2
        # But original code has j=1 and k+=j for each iteration, so k = iteration+1
        k_vals = current_offsets + 1
        
        # Calculate b indices: LEN_1D - k
        b_indices = LEN_1D - k_vals
        b_mask = i_mask & (b_indices >= 0) & (b_indices < LEN_1D)
        
        # Load values
        a_vals = tl.load(a_ptr + i_vals, mask=b_mask, other=0.0)
        b_vals = tl.load(b_ptr + b_indices, mask=b_mask, other=0.0)
        
        # Update a
        result = a_vals + b_vals
        tl.store(a_ptr + i_vals, result, mask=b_mask)

def s122_triton(a, b, n1, n3):
    LEN_1D = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate number of iterations
    num_iterations = (LEN_1D - (n1 - 1) + n3 - 1) // n3
    
    if num_iterations > 0:
        grid = (1,)
        s122_kernel[grid](a, b, n1, n3, LEN_1D, BLOCK_SIZE)