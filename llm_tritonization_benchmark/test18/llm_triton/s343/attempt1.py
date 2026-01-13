import torch
import triton
import triton.language as tl

@triton.jit
def s343_kernel(aa_ptr, bb_ptr, flat_2d_array_ptr, n_2d):
    # This is a stream compaction pattern - pack elements based on condition
    # Must be done sequentially due to counter dependencies
    # Use single thread to maintain correctness
    
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    k = 0  # Counter for packed elements
    
    for i in range(n_2d):
        for j in range(n_2d):
            # Load bb[j][i] and aa[j][i]
            bb_idx = j * n_2d + i
            aa_idx = j * n_2d + i
            
            bb_val = tl.load(bb_ptr + bb_idx)
            
            if bb_val > 0.0:
                aa_val = tl.load(aa_ptr + aa_idx)
                tl.store(flat_2d_array_ptr + k, aa_val)
                k += 1

def s343_triton(aa, bb, flat_2d_array):
    n_2d = aa.shape[0]
    
    # Launch single thread kernel to handle stream compaction sequentially
    grid = (1,)
    
    s343_kernel[grid](
        aa, bb, flat_2d_array,
        n_2d
    )