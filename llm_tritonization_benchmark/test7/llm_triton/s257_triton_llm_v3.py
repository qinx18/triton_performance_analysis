import torch
import triton
import triton.language as tl

@triton.jit
def s257_kernel(a_ptr, a_copy_ptr, aa_ptr, bb_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(1, LEN_2D):
        i_offsets = block_start + offsets
        mask = i_offsets < LEN_2D
        
        # Load a[i-1] values
        prev_i_idx = i - 1
        a_prev_vals = tl.load(a_copy_ptr + prev_i_idx, mask=mask)
        
        for block_j in range(0, LEN_2D, BLOCK_SIZE):
            j_offsets = block_j + offsets
            j_mask = j_offsets < LEN_2D
            
            # Calculate 2D indices for aa and bb
            aa_idx = j_offsets * LEN_2D + i
            bb_idx = j_offsets * LEN_2D + i
            
            # Load aa[j][i] and bb[j][i] values
            aa_vals = tl.load(aa_ptr + aa_idx, mask=j_mask)
            bb_vals = tl.load(bb_ptr + bb_idx, mask=j_mask)
            
            # Compute a[i] = aa[j][i] - a[i-1]
            a_new = aa_vals - a_prev_vals
            
            # Store a[i]
            tl.store(a_ptr + i, a_new, mask=j_mask)
            
            # Compute aa[j][i] = a[i] + bb[j][i]
            aa_new = a_new + bb_vals
            
            # Store aa[j][i]
            tl.store(aa_ptr + aa_idx, aa_new, mask=j_mask)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Create read-only copy for WAR dependency
    a_copy = a.clone()
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s257_kernel[grid](
        a, a_copy, aa, bb, LEN_2D, BLOCK_SIZE
    )