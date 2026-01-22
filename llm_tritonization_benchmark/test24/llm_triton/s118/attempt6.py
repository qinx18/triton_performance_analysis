import triton
import triton.language as tl
import torch

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i_val, BLOCK_SIZE: tl.constexpr):
    j_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Only process valid j values: 0 <= j <= i-1
    mask = j_idx < i_val
    
    # Load bb[j, i] values
    bb_offsets = j_idx * 256 + i_val  # bb[j][i] = j * LEN_2D + i
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask, other=0.0)
    
    # Load a[i-j-1] values
    a_read_indices = i_val - j_idx - 1
    a_vals = tl.load(a_ptr + a_read_indices, mask=mask, other=0.0)
    
    # Compute products
    products = bb_vals * a_vals
    
    # Sum all products for this i
    result = tl.sum(products)
    
    # Only one thread writes the result
    if tl.program_id(0) == 0:
        # Load current a[i] and add the sum
        current_val = tl.load(a_ptr + i_val)
        tl.store(a_ptr + i_val, current_val + result)

def s118_triton(a, bb):
    LEN_2D = bb.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential loop over i
    for i in range(1, LEN_2D):
        num_j = i  # j goes from 0 to i-1, so i values total
        grid = (triton.cdiv(num_j, BLOCK_SIZE),)
        s118_kernel[grid](a, bb, i, BLOCK_SIZE=BLOCK_SIZE)