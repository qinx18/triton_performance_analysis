import torch
import triton
import triton.language as tl

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i_val, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for j dimension
    pid = tl.program_id(0)
    
    # Calculate j offsets for this block
    j_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid j values: j <= i - 1
    j_mask = j_offsets < i_val
    
    # Load bb[j][i] values
    bb_offsets = j_offsets * LEN_2D + i_val
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
    
    # Load a[i-j-1] values
    a_read_indices = i_val - j_offsets - 1
    a_read_mask = j_mask & (a_read_indices >= 0)
    a_vals = tl.load(a_ptr + a_read_indices, mask=a_read_mask, other=0.0)
    
    # Compute products
    products = bb_vals * a_vals
    
    # Mask out invalid j values
    products = tl.where(j_mask, products, 0.0)
    
    # Sum across all valid j values in this block
    block_sum = tl.sum(products)
    
    # Atomic add to a[i]
    tl.atomic_add(a_ptr + i_val, block_sum)

def s118_triton(a, bb):
    LEN_2D = 256
    BLOCK_SIZE = 64
    
    # Sequential loop over i
    for i in range(1, LEN_2D):
        # Number of j values for this i: j from 0 to i-1
        num_j = i
        
        if num_j > 0:
            # Calculate grid size for j dimension
            grid_size = triton.cdiv(num_j, BLOCK_SIZE)
            
            # Launch kernel for this i value
            s118_kernel[(grid_size,)](
                a, bb, i, LEN_2D, BLOCK_SIZE
            )