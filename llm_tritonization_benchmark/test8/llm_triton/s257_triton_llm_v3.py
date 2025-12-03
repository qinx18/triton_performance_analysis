import torch
import triton
import triton.language as tl

@triton.jit
def s257_kernel(a_ptr, a_copy_ptr, aa_ptr, bb_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(1, LEN_2D):
        j_start = tl.program_id(0) * BLOCK_SIZE
        j_indices = j_start + j_offsets
        j_mask = j_indices < LEN_2D
        
        # Load a[i-1] (scalar broadcast)
        a_prev_offset = i - 1
        a_prev = tl.load(a_copy_ptr + a_prev_offset)
        
        # Load aa[j][i] values
        aa_offsets = j_indices * LEN_2D + i
        aa_vals = tl.load(aa_ptr + aa_offsets, mask=j_mask)
        
        # Compute a[i] = aa[j][i] - a[i-1]
        a_new = aa_vals - a_prev
        
        # Store a[i] (broadcast to all valid j positions, but only store once)
        if j_start == 0:  # Only first block stores a[i]
            tl.store(a_ptr + i, a_new[0])
        
        # Load bb[j][i] values
        bb_offsets = j_indices * LEN_2D + i
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask)
        
        # Compute aa[j][i] = a[i] + bb[j][i]
        aa_new = a_new + bb_vals
        
        # Store aa[j][i]
        tl.store(aa_ptr + aa_offsets, aa_new, mask=j_mask)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 32
    
    # Create read-only copy of a to handle WAR dependency
    a_copy = a.clone()
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s257_kernel[grid](
        a, a_copy, aa, bb, 
        LEN_2D, BLOCK_SIZE
    )