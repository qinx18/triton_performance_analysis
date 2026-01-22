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
    
    # Reduce to scalar and accumulate
    partial_sum = tl.sum(products)
    
    # All threads in first block contribute to atomic add
    if tl.program_id(0) == 0:
        # Use reduction across threads
        total_sum = tl.sum(partial_sum)
        # Only first thread writes
        if (tl.arange(0, BLOCK_SIZE) == 0)[0]:
            tl.atomic_add(a_ptr + i_val, total_sum)
    else:
        # Other blocks just do atomic add of their partial sum
        if (tl.arange(0, BLOCK_SIZE) == 0)[0]:
            tl.atomic_add(a_ptr + i_val, partial_sum)

def s118_triton(a, bb):
    LEN_2D = bb.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential loop over i
    for i in range(1, LEN_2D):
        num_j = i  # j goes from 0 to i-1, so i values total
        grid = (triton.cdiv(num_j, BLOCK_SIZE),)
        s118_kernel[grid](a, bb, i, BLOCK_SIZE=BLOCK_SIZE)