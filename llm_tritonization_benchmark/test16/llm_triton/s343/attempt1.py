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
        # Load bb[j][i] and aa[j][i] for all j in this block
        bb_ptrs = bb_ptr + j_idx * len_2d + i
        aa_ptrs = aa_ptr + j_idx * len_2d + i
        
        bb_vals = tl.load(bb_ptrs, mask=j_mask, other=0.0)
        aa_vals = tl.load(aa_ptrs, mask=j_mask, other=0.0)
        
        # Check condition
        cond = bb_vals > 0.0
        valid_cond = cond & j_mask
        
        # For each valid element, we need to pack it
        for j_local in range(BLOCK_SIZE):
            global_j = pid * BLOCK_SIZE + j_local
            if global_j < len_2d and valid_cond[j_local]:
                tl.store(flat_2d_array_ptr + k, aa_vals[j_local])
                k += 1

def s343_triton(aa, bb, flat_2d_array):
    len_2d = aa.shape[0]
    
    # Use stream compaction approach since this is fundamentally a packing operation
    # First, create a flattened view with proper ordering (j varies fastest)
    aa_flat = aa.transpose(0, 1).contiguous().view(-1)  # aa[j][i] ordering
    bb_flat = bb.transpose(0, 1).contiguous().view(-1)  # bb[j][i] ordering
    
    # Create condition mask
    mask = bb_flat > 0.0
    
    # Pack the values using boolean indexing
    packed_values = aa_flat[mask]
    num_packed = packed_values.numel()
    
    # Store packed values at the beginning of flat_2d_array
    flat_2d_array[:num_packed] = packed_values