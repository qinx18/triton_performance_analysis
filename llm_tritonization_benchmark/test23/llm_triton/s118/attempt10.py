import torch
import triton
import triton.language as tl

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i_val, N, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    total_sum = 0.0
    
    # Process all j values for this i in chunks
    num_blocks = (i_val + BLOCK_SIZE - 1) // BLOCK_SIZE
    for block_idx in range(num_blocks):
        j_start = block_idx * BLOCK_SIZE
        current_j = j_start + j_offsets
        j_mask = current_j < i_val
        
        # Load bb[j][i] values
        bb_indices = current_j * N + i_val
        bb_vals = tl.load(bb_ptr + bb_indices, mask=j_mask, other=0.0)
        
        # Load a[i-j-1] values
        a_read_indices = i_val - current_j - 1
        a_read_mask = j_mask & (a_read_indices >= 0)
        a_vals = tl.load(a_ptr + a_read_indices, mask=a_read_mask, other=0.0)
        
        # Compute products and accumulate
        products = tl.where(a_read_mask, bb_vals * a_vals, 0.0)
        total_sum += tl.sum(products)
    
    # Add to a[i]
    current_a = tl.load(a_ptr + i_val)
    tl.store(a_ptr + i_val, current_a + total_sum)

def s118_triton(a, bb):
    N = bb.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential loop over i
    for i in range(1, N):
        grid = (1,)
        s118_kernel[grid](a, bb, i, N, BLOCK_SIZE)
    
    return a