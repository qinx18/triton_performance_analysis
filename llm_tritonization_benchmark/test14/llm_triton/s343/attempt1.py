import torch
import triton
import triton.language as tl

@triton.jit
def s343_kernel(aa_ptr, bb_ptr, flat_2d_array_ptr, len_2d):
    # This is a stream compaction pattern that's difficult to parallelize efficiently
    # We'll use a single thread approach since the output indexing depends on
    # how many previous elements passed the condition
    
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    k = 0
    
    for i in range(len_2d):
        for j in range(len_2d):
            # Load bb[j][i] and aa[j][i]
            bb_offset = j * len_2d + i
            aa_offset = j * len_2d + i
            
            bb_val = tl.load(bb_ptr + bb_offset)
            
            if bb_val > 0.0:
                aa_val = tl.load(aa_ptr + aa_offset)
                tl.store(flat_2d_array_ptr + k, aa_val)
                k += 1

def s343_triton(aa, bb, flat_2d_array):
    len_2d = aa.shape[0]
    
    # Launch single thread kernel since this is inherently sequential
    grid = (1,)
    
    s343_kernel[grid](
        aa, bb, flat_2d_array,
        len_2d
    )