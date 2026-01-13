import torch
import triton
import triton.language as tl

@triton.jit
def s141_kernel(flat_2d_array_ptr, bb_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i = pid
    
    if i >= n:
        return
    
    # k = (i+1) * ((i+1) - 1) / 2 + (i+1)-1
    # Simplifying: k = i * (i + 1) / 2 + i = i * (i + 3) / 2
    k_start = i * (i + 1) // 2 + i
    
    # Process j from i to n-1
    num_j = n - i
    
    # Process in blocks
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    k_current = k_start
    for j_block in range(0, num_j, BLOCK_SIZE):
        current_j = i + j_block + j_offsets
        mask = (j_block + j_offsets) < num_j
        
        # Calculate k indices for this block
        # k increases by j+1 at each step
        k_indices = k_current + tl.cumsum(current_j, axis=0) - tl.sum(tl.where(j_offsets == 0, current_j, 0))
        
        # Load bb[j][i] values
        bb_indices = current_j * n + i
        bb_vals = tl.load(bb_ptr + bb_indices, mask=mask, other=0.0)
        
        # Load current flat_2d_array values
        flat_vals = tl.load(flat_2d_array_ptr + k_indices, mask=mask, other=0.0)
        
        # Update
        new_vals = flat_vals + bb_vals
        tl.store(flat_2d_array_ptr + k_indices, new_vals, mask=mask)
        
        # Update k_current for next block
        if j_block + BLOCK_SIZE < num_j:
            k_current += tl.sum(current_j + 1, axis=0)

def s141_triton(bb, flat_2d_array):
    n = bb.shape[0]
    
    BLOCK_SIZE = 64
    grid = (n,)
    
    s141_kernel[grid](
        flat_2d_array, bb, n, BLOCK_SIZE
    )