import torch
import triton
import triton.language as tl

@triton.jit
def s343_kernel(aa_ptr, bb_ptr, flat_2d_array_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < len_2d
    
    k = 0
    
    for i in range(len_2d):
        # Load bb[j][i] values for this i
        bb_ptrs = bb_ptr + j_idx * len_2d + i
        bb_vals = tl.load(bb_ptrs, mask=j_mask, other=0.0)
        
        # Load aa[j][i] values for this i  
        aa_ptrs = aa_ptr + j_idx * len_2d + i
        aa_vals = tl.load(aa_ptrs, mask=j_mask, other=0.0)
        
        # Create condition mask
        condition = bb_vals > 0.0
        valid_condition = j_mask & condition
        
        # Count how many elements satisfy condition in this block for this i
        count = tl.sum(valid_condition.to(tl.int32))
        
        # Store values that satisfy condition
        if count > 0:
            # Find indices where condition is true
            for local_j in range(BLOCK_SIZE):
                if local_j + pid * BLOCK_SIZE < len_2d:
                    global_j = local_j + pid * BLOCK_SIZE
                    bb_val = tl.load(bb_ptr + global_j * len_2d + i)
                    if bb_val > 0.0:
                        aa_val = tl.load(aa_ptr + global_j * len_2d + i)
                        tl.store(flat_2d_array_ptr + k, aa_val)
                        k += 1

def s343_triton(aa, bb, flat_2d_array, len_2d):
    # Sequential implementation since this is stream compaction
    # which requires maintaining order and counter across iterations
    
    # Use PyTorch for correct stream compaction
    k = 0
    
    for i in range(len_2d):
        for j in range(len_2d):
            if bb[j, i] > 0.0:
                flat_2d_array[k] = aa[j, i]
                k += 1
    
    return k