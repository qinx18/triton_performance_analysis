import triton
import triton.language as tl
import torch

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i_val, BLOCK_SIZE: tl.constexpr):
    # Get the number of j values to process (0 to i-1, so i total values)
    n_j = i_val
    
    # Get block of j indices
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < n_j
    
    # Load bb[j][i] values - bb is row-major, so bb[j][i] = bb[j * N + i]
    bb_offsets = j_offsets * 256 + i_val  # assuming 256x256 matrix
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
    
    # Load a[i-j-1] values
    a_indices = i_val - j_offsets - 1
    a_vals = tl.load(a_ptr + a_indices, mask=j_mask, other=0.0)
    
    # Compute bb[j][i] * a[i-j-1]
    products = bb_vals * a_vals
    
    # Sum all products
    result = tl.sum(products)
    
    # Load current a[i] and add the sum
    current_a = tl.load(a_ptr + i_val)
    new_a = current_a + result
    
    # Store back to a[i]
    tl.store(a_ptr + i_val, new_a)

def s118_triton(a, bb):
    # Get dimensions from input tensors
    LEN_2D = bb.shape[0]
    
    # Sequential loop over i, parallel processing of j
    BLOCK_SIZE = 256
    
    for i in range(1, LEN_2D):
        # Launch kernel for this i value
        grid = (1,)  # Only one block needed since we process all j values in one kernel
        s118_kernel[grid](
            a, bb, i, 
            BLOCK_SIZE=BLOCK_SIZE
        )