import triton
import triton.language as tl
import torch

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i_val, n_cols, BLOCK_SIZE: tl.constexpr):
    # Get the number of j values to process (0 to i-1, so i total values)
    n_j = i_val
    
    # Get block of j indices
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < n_j
    
    # Load bb[j][i] values - bb is row-major, so bb[j][i] = bb[j * n_cols + i]
    bb_offsets = j_offsets * n_cols + i_val
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
    
    # Load a[i-j-1] values
    a_indices = i_val - j_offsets - 1
    # Need to ensure a_indices are valid (>= 0)
    a_mask = j_mask & (a_indices >= 0)
    a_vals = tl.load(a_ptr + a_indices, mask=a_mask, other=0.0)
    
    # Compute bb[j][i] * a[i-j-1] only where both masks are valid
    products = tl.where(a_mask, bb_vals * a_vals, 0.0)
    
    # Sum all products
    result = tl.sum(products)
    
    # Atomically add the result to a[i]
    tl.atomic_add(a_ptr + i_val, result)

def s118_triton(a, bb):
    # Get dimensions from input tensors
    LEN_2D = bb.shape[0]
    n_cols = bb.shape[1]
    
    # Sequential loop over i, parallel processing of j
    BLOCK_SIZE = 256
    
    for i in range(1, LEN_2D):
        # Launch kernel for this i value
        grid = (triton.cdiv(i, BLOCK_SIZE),)
        s118_kernel[grid](
            a, bb, i, n_cols,
            BLOCK_SIZE=BLOCK_SIZE
        )