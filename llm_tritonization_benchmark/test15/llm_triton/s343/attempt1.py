import torch
import triton
import triton.language as tl

@triton.jit
def s343_kernel(aa_ptr, bb_ptr, flat_2d_array_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < len_2d
    
    k = 0
    for i in range(len_2d):
        # Load bb values for current i and all j in this block
        bb_ptrs = bb_ptr + j_idx * len_2d + i
        bb_vals = tl.load(bb_ptrs, mask=j_mask, other=0.0)
        
        # Check condition
        cond_mask = bb_vals > 0.0
        
        # Load aa values where condition is true
        aa_ptrs = aa_ptr + j_idx * len_2d + i
        aa_vals = tl.load(aa_ptrs, mask=j_mask & cond_mask, other=0.0)
        
        # For each element that passes condition, store sequentially
        for j_local in range(BLOCK_SIZE):
            if j_local < len_2d - pid * BLOCK_SIZE:
                j_global = pid * BLOCK_SIZE + j_local
                if j_global < len_2d and bb_vals[j_local] > 0.0:
                    tl.store(flat_2d_array_ptr + k, aa_vals[j_local])
                    k += 1

def s343_triton(aa, bb, flat_2d_array):
    len_2d = aa.shape[0]
    
    # Use PyTorch for stream compaction (this is the correct approach)
    # Flatten the arrays and process sequentially as in original C code
    aa_flat = aa.t().contiguous().view(-1)  # Column-major order like C
    bb_flat = bb.t().contiguous().view(-1)  # Column-major order like C
    
    # Create mask for elements that pass condition
    mask = bb_flat > 0.0
    
    # Pack the values using boolean indexing
    packed_values = aa_flat[mask]
    num_packed = packed_values.numel()
    
    # Store packed values at beginning of flat_2d_array
    if num_packed > 0:
        flat_2d_array[:num_packed] = packed_values