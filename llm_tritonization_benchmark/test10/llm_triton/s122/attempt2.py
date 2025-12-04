import triton
import triton.language as tl
import torch

@triton.jit
def s122_kernel(a_ptr, b_ptr, n1, n3, LEN_1D, BLOCK_SIZE: tl.constexpr):
    # Calculate the number of iterations in the sequential loop
    num_iters = (LEN_1D - (n1 - 1) + n3 - 1) // n3
    
    # Get block ID and calculate which iterations this block handles
    block_id = tl.program_id(0)
    iter_offsets = tl.arange(0, BLOCK_SIZE)
    iter_indices = block_id * BLOCK_SIZE + iter_offsets
    iter_mask = iter_indices < num_iters
    
    # For each valid iteration, compute the values
    i_values = (n1 - 1) + iter_indices * n3
    k_values = iter_indices + 1  # k accumulates: 1, 2, 3, ...
    
    # Load a[i] and b[LEN_1D - k] for valid iterations
    a_vals = tl.load(a_ptr + i_values, mask=iter_mask, other=0.0)
    b_vals = tl.load(b_ptr + (LEN_1D - k_values), mask=iter_mask, other=0.0)
    
    # Update a[i]
    new_vals = a_vals + b_vals
    tl.store(a_ptr + i_values, new_vals, mask=iter_mask)

def s122_triton(a, b, n1, n3):
    LEN_1D = a.shape[0]
    
    # Calculate number of iterations
    num_iters = (LEN_1D - (n1 - 1) + n3 - 1) // n3
    
    if num_iters <= 0:
        return
    
    BLOCK_SIZE = 128
    grid = (triton.cdiv(num_iters, BLOCK_SIZE),)
    
    s122_kernel[grid](
        a, b,
        n1, n3, LEN_1D,
        BLOCK_SIZE=BLOCK_SIZE
    )