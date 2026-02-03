import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, len_2d: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < len_2d
    
    for i in range(1, len_2d):
        # Load a[i-1] (scalar)
        a_prev = tl.load(a_ptr + (i - 1))
        
        # Sequential processing through j to handle overwrites correctly
        current_a_i = 0.0
        for j_block_start in range(0, len_2d, BLOCK_SIZE):
            j_indices = j_block_start + j_offsets
            j_valid = (j_indices < len_2d) & (j_indices >= 0)
            
            if j_block_start < len_2d:
                # Load aa[j][i] for current block
                aa_ptrs = aa_ptr + j_indices * len_2d + i
                aa_vals = tl.load(aa_ptrs, mask=j_valid, other=0.0)
                
                # Process each j in sequence within the block
                for j_offset in range(min(BLOCK_SIZE, len_2d - j_block_start)):
                    j = j_block_start + j_offset
                    if j < len_2d:
                        # Extract aa[j][i] value
                        is_current_j = j_indices == j
                        aa_j_i = tl.sum(tl.where(is_current_j & j_valid, aa_vals, 0.0))
                        
                        # Compute a[i] = aa[j][i] - a[i-1]
                        current_a_i = aa_j_i - a_prev
                        
                        # Store a[i]
                        tl.store(a_ptr + i, current_a_i)
        
        # Now update all aa[j][i] with the final a[i] value
        final_a_i = tl.load(a_ptr + i)
        
        for j_block_start in range(0, len_2d, BLOCK_SIZE):
            j_indices = j_block_start + j_offsets
            j_valid = (j_indices < len_2d) & (j_indices >= 0)
            
            if j_block_start < len_2d:
                # Load bb[j][i]
                bb_ptrs = bb_ptr + j_indices * len_2d + i
                bb_vals = tl.load(bb_ptrs, mask=j_valid, other=0.0)
                
                # Compute aa[j][i] = a[i] + bb[j][i]
                aa_new_vals = final_a_i + bb_vals
                
                # Store aa[j][i]
                aa_ptrs = aa_ptr + j_indices * len_2d + i
                tl.store(aa_ptrs, aa_new_vals, mask=j_valid)

def s257_triton(a, aa, bb, len_2d):
    BLOCK_SIZE = triton.next_power_of_2(len_2d)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)
    
    grid = (1,)
    
    s257_kernel[grid](
        a, aa, bb, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )