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
        
        # Process each j sequentially to ensure correct overwrite behavior
        for j_start in range(0, len_2d, BLOCK_SIZE):
            j_idx = j_start + j_offsets
            j_valid = j_idx < len_2d
            
            if j_start < len_2d:
                # Load aa[j][i] for current block of j values
                aa_ji_ptrs = aa_ptr + j_idx * len_2d + i
                aa_ji = tl.load(aa_ji_ptrs, mask=j_valid)
                
                # Compute new a[i] values for each j in this block
                a_i_candidates = aa_ji - a_prev
                
                # Find the last valid j in this block and use its a[i] value
                last_valid_j_in_block = tl.max(tl.where(j_valid, j_idx, -1))
                
                # Only update a[i] if this block contains valid j values
                if j_start == 0 or last_valid_j_in_block >= 0:
                    # Extract the a[i] value from the last valid j in this block
                    is_last_j = j_idx == last_valid_j_in_block
                    a_i_from_last_j = tl.sum(tl.where(is_last_j & j_valid, a_i_candidates, 0.0))
                    
                    # Store a[i] (will be overwritten by subsequent blocks)
                    if tl.sum(is_last_j.to(tl.int32)) > 0:
                        tl.store(a_ptr + i, a_i_from_last_j)
        
        # Now load the final a[i] value and update aa for all j
        a_i_final = tl.load(a_ptr + i)
        
        # Update aa[j][i] = a[i] + bb[j][i] for all j in parallel
        for j_start in range(0, len_2d, BLOCK_SIZE):
            j_idx = j_start + j_offsets
            j_valid = j_idx < len_2d
            
            if j_start < len_2d:
                # Load bb[j][i]
                bb_ji_ptrs = bb_ptr + j_idx * len_2d + i
                bb_ji = tl.load(bb_ji_ptrs, mask=j_valid)
                
                # Compute aa[j][i] = a[i] + bb[j][i]
                aa_ji_new = a_i_final + bb_ji
                
                # Store aa[j][i]
                aa_ji_ptrs = aa_ptr + j_idx * len_2d + i
                tl.store(aa_ji_ptrs, aa_ji_new, mask=j_valid)

def s257_triton(a, aa, bb, len_2d):
    BLOCK_SIZE = triton.next_power_of_2(len_2d)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)
    
    grid = (1,)
    
    s257_kernel[grid](
        a, aa, bb, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )