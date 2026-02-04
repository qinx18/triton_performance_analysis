import torch
import triton
import triton.language as tl

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Each block handles a different i value
    i = pid + 1  # Start from i=1
    
    if i >= len_2d:
        return
    
    # Load a[i-1] (scalar)
    a_prev = tl.load(a_ptr + (i - 1))
    
    # Parallelize over j dimension
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < len_2d
    
    # For each j, compute the operations sequentially to handle overwrites
    for j_start in range(0, len_2d, BLOCK_SIZE):
        current_j = j_start + j_offsets
        current_mask = (current_j < len_2d) & j_mask
        
        if not tl.any(current_mask):
            break
            
        # Load aa[j][i] for all valid j in this block
        aa_ptrs = aa_ptr + current_j * len_2d + i
        aa_vals = tl.load(aa_ptrs, mask=current_mask)
        
        # Load bb[j][i] for all valid j in this block
        bb_ptrs = bb_ptr + current_j * len_2d + i
        bb_vals = tl.load(bb_ptrs, mask=current_mask)
        
        # Process each j sequentially within this block to handle overwrites
        for local_idx in range(BLOCK_SIZE):
            if j_start + local_idx < len_2d:
                j = j_start + local_idx
                
                # Load aa[j][i]
                aa_val = tl.load(aa_ptr + j * len_2d + i)
                
                # Compute a[i] = aa[j][i] - a[i-1]
                a_new = aa_val - a_prev
                tl.store(a_ptr + i, a_new)
                
                # Load bb[j][i]
                bb_val = tl.load(bb_ptr + j * len_2d + i)
                
                # Compute aa[j][i] = a[i] + bb[j][i]
                aa_new = a_new + bb_val
                tl.store(aa_ptr + j * len_2d + i, aa_new)

def s257_triton(a, aa, bb, len_2d):
    BLOCK_SIZE = 256
    
    # Launch one block per i value (i=1 to len_2d-1)
    num_i_values = len_2d - 1
    grid = (num_i_values,)
    
    s257_kernel[grid](
        a, aa, bb, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )