import triton
import triton.language as tl
import torch

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i_val, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID for j dimension
    pid = tl.program_id(0)
    
    # Calculate j offsets for this block
    j_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Mask to ensure j <= i - 1
    j_mask = j_offsets < i_val
    
    # Load bb[j][i] values - bb is row-major, so bb[j][i] = bb_ptr[j * N + i]
    bb_offsets = j_offsets * N + i_val
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
    
    # Load a[i-j-1] values
    a_read_indices = i_val - j_offsets - 1
    a_read_mask = j_mask & (a_read_indices >= 0) & (a_read_indices < N)
    a_vals = tl.load(a_ptr + a_read_indices, mask=a_read_mask, other=0.0)
    
    # Compute products
    products = bb_vals * a_vals
    masked_products = tl.where(a_read_mask, products, 0.0)
    
    # Sum reduction within each block
    result = tl.sum(masked_products)
    
    # Only the first thread in valid blocks writes the result
    if pid * BLOCK_SIZE < i_val and tl.program_id(1) == 0 and tl.program_id(2) == 0:
        tl.atomic_add(a_ptr + i_val, result)

def s118_triton(a, bb):
    N = bb.shape[0]
    BLOCK_SIZE = 64
    
    # Sequential loop over i
    for i in range(1, N):
        num_j = i
        grid = (triton.cdiv(num_j, BLOCK_SIZE),)
        
        s118_kernel[grid](
            a, bb, i, N,
            BLOCK_SIZE=BLOCK_SIZE
        )