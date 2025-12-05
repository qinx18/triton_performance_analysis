import triton
import triton.language as tl
import torch

@triton.jit
def s122_kernel(a_ptr, b_ptr, n_elements, n1, n3, LEN_1D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    global_offsets = block_start + offsets
    
    mask = global_offsets < n_elements
    
    # For each element in the block, compute the corresponding i and k values
    i_vals = n1 - 1 + global_offsets * n3
    k_vals = global_offsets + 1  # k starts at 0 and increments by j=1 each iteration
    
    # Compute b indices
    b_indices = LEN_1D - k_vals
    
    # Create masks for valid operations
    i_mask = i_vals < LEN_1D
    b_mask = b_indices >= 0
    valid_mask = mask & i_mask & b_mask
    
    # Load values where valid
    a_vals = tl.load(a_ptr + i_vals, mask=valid_mask, other=0.0)
    b_vals = tl.load(b_ptr + b_indices, mask=valid_mask, other=0.0)
    
    # Compute and store result
    result = a_vals + b_vals
    tl.store(a_ptr + i_vals, result, mask=valid_mask)

def s122_triton(a, b, n1, n3):
    LEN_1D = a.shape[0]
    
    # Calculate number of iterations in the original loop
    n_elements = 0
    i = n1 - 1
    while i < LEN_1D:
        n_elements += 1
        i += n3
    
    if n_elements == 0:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s122_kernel[grid](
        a, b, n_elements, n1, n3, LEN_1D, BLOCK_SIZE
    )