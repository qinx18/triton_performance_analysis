import torch
import triton
import triton.language as tl

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i_val, BLOCK_SIZE: tl.constexpr):
    # Get program ID for j dimension
    pid = tl.program_id(0)
    
    # Calculate j values for this block
    j_start = pid * BLOCK_SIZE
    j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid j values (j <= i-1)
    j_mask = j_offsets <= (i_val - 1)
    
    # Load bb[j][i] values
    bb_offsets = j_offsets * 256 + i_val  # bb is 256x256
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
    
    # Load a[i-j-1] values
    a_indices = i_val - j_offsets - 1
    a_mask = j_mask & (a_indices >= 0)
    a_vals = tl.load(a_ptr + a_indices, mask=a_mask, other=0.0)
    
    # Compute products and sum
    products = bb_vals * a_vals
    products = tl.where(a_mask, products, 0.0)
    result = tl.sum(products)
    
    # Atomically add to a[i]
    if pid == 0:  # Only one thread should update
        tl.atomic_add(a_ptr + i_val, result)

def s118_triton(a, bb):
    LEN_2D = bb.shape[0]
    BLOCK_SIZE = 64
    
    # Sequential loop over i
    for i in range(1, LEN_2D):
        # Number of j values (0 to i-1)
        num_j = i
        
        if num_j > 0:
            # Launch kernel for this i
            grid = (triton.cdiv(num_j, BLOCK_SIZE),)
            s118_kernel[grid](a, bb, i, BLOCK_SIZE=BLOCK_SIZE)
    
    return a