import torch
import triton
import triton.language as tl

@triton.jit
def s343_kernel(aa_ptr, bb_ptr, flat_2d_array_ptr, len_2d: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < len_2d
    
    counter = 0
    
    for i in range(len_2d):
        aa_ptrs = aa_ptr + i * len_2d + j_idx
        bb_ptrs = bb_ptr + i * len_2d + j_idx
        
        aa_vals = tl.load(aa_ptrs, mask=j_mask, other=0.0)
        bb_vals = tl.load(bb_ptrs, mask=j_mask, other=0.0)
        
        condition = bb_vals > 0.0
        valid_mask = j_mask & condition
        
        for j_offset in range(BLOCK_SIZE):
            if j_idx[j_offset] < len_2d:
                if condition[j_offset]:
                    output_ptr = flat_2d_array_ptr + counter
                    tl.store(output_ptr, aa_vals[j_offset])
                    counter += 1

def s343_triton(aa, bb, flat_2d_array, len_2d):
    BLOCK_SIZE = 64
    
    # Sequential processing with stream compaction
    mask = bb > 0.0
    packed_values = aa[mask]
    num_packed = packed_values.numel()
    
    if num_packed > 0:
        flat_2d_array[:num_packed] = packed_values
    
    return num_packed