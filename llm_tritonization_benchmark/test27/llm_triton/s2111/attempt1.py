import triton
import triton.language as tl
import torch

@triton.jit
def s2111_kernel(aa_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Sequential processing of diagonals
    # Each diagonal k contains elements where j + i = k
    for k in range(2, 2 * n):  # k = j + i, from 2 to 2*n-1
        # Calculate range of j values for this diagonal
        start_j = tl.maximum(1, k - n + 1)
        end_j = tl.minimum(k, n)
        
        # Process elements on this diagonal in blocks
        for block_start in range(start_j, end_j, BLOCK_SIZE):
            block_end = tl.minimum(block_start + BLOCK_SIZE, end_j)
            
            # Get j values for this block
            j_offsets = tl.arange(0, BLOCK_SIZE)
            j_vals = block_start + j_offsets
            
            # Calculate corresponding i values: i = k - j
            i_vals = k - j_vals
            
            # Create mask for valid elements
            mask = (j_vals < block_end) & (j_vals >= start_j) & (i_vals >= 1) & (i_vals < n)
            
            if tl.sum(mask.to(tl.int32)) > 0:
                # Calculate array indices
                curr_idx = j_vals * n + i_vals
                left_idx = j_vals * n + (i_vals - 1)
                up_idx = (j_vals - 1) * n + i_vals
                
                # Load values
                left_vals = tl.load(aa_ptr + left_idx, mask=mask)
                up_vals = tl.load(aa_ptr + up_idx, mask=mask)
                
                # Compute new values
                new_vals = (left_vals + up_vals) / 1.9
                
                # Store results
                tl.store(aa_ptr + curr_idx, new_vals, mask=mask)

def s2111_triton(aa, len_2d):
    n = len_2d
    BLOCK_SIZE = 32
    
    # Launch kernel with grid size 1 since we process sequentially
    grid = (1,)
    s2111_kernel[grid](aa, n, BLOCK_SIZE)
    
    return aa