import triton
import triton.language as tl
import torch

@triton.jit
def s122_kernel(a_ptr, b_ptr, n1, n3, len_1d, BLOCK_SIZE: tl.constexpr):
    # Calculate iteration space
    max_iter = (len_1d - (n1 - 1) + n3 - 1) // n3
    
    # Block-level iteration
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    iter_ids = block_start + offsets
    
    mask = iter_ids < max_iter
    
    # Calculate i indices: i = n1-1 + iter_id * n3
    i_indices = (n1 - 1) + iter_ids * n3
    
    # Calculate k values: k = 1 + iter_id (since j=1 and k += j for each iteration)
    k_values = 1 + iter_ids
    
    # Calculate b indices: LEN_1D - k
    b_indices = len_1d - k_values
    
    # Load values with masking
    a_vals = tl.load(a_ptr + i_indices, mask=mask)
    b_vals = tl.load(b_ptr + b_indices, mask=mask)
    
    # Perform computation
    result = a_vals + b_vals
    
    # Store result
    tl.store(a_ptr + i_indices, result, mask=mask)

def s122_triton(a, b, n1, n3):
    len_1d = a.shape[0]
    
    # Calculate total iterations needed
    max_iter = (len_1d - (n1 - 1) + n3 - 1) // n3
    
    if max_iter <= 0:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(max_iter, BLOCK_SIZE),)
    
    s122_kernel[grid](
        a, b, n1, n3, len_1d,
        BLOCK_SIZE=BLOCK_SIZE
    )