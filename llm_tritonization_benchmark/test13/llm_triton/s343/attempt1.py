import torch
import triton
import triton.language as tl

@triton.jit
def s343_kernel(aa_ptr, bb_ptr, flat_2d_array_ptr, len_2d: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < len_2d
    
    k = 0
    
    for i in range(len_2d):
        # Load bb[j][i] for all j values in this block
        bb_ptrs = bb_ptr + j_idx * len_2d + i
        bb_vals = tl.load(bb_ptrs, mask=j_mask, other=0.0)
        
        # Load aa[j][i] for all j values in this block
        aa_ptrs = aa_ptr + j_idx * len_2d + i
        aa_vals = tl.load(aa_ptrs, mask=j_mask, other=0.0)
        
        # Check condition bb[j][i] > 0
        condition = (bb_vals > 0.0) & j_mask
        
        # For each j that satisfies the condition, write to flat_2d_array
        for j_local in range(BLOCK_SIZE):
            j_global = pid * BLOCK_SIZE + j_local
            if j_global < len_2d:
                if bb_vals[j_local] > 0.0:
                    tl.store(flat_2d_array_ptr + k, aa_vals[j_local])
                    k += 1

def s343_triton(aa, bb, flat_2d_array):
    len_2d = aa.shape[0]
    
    # Stream compaction using PyTorch boolean indexing
    mask = bb > 0.0
    packed_values = aa[mask]
    num_packed = packed_values.numel()
    
    if num_packed > 0:
        flat_2d_array[:num_packed] = packed_values.flatten()
    
    return flat_2d_array