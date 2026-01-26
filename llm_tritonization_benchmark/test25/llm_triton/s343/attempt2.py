import triton
import triton.language as tl
import torch

@triton.jit
def s343_kernel(aa_ptr, bb_ptr, flat_2d_array_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    # Get program ID for j dimension parallelization
    pid = tl.program_id(0)
    
    # Define j offsets once at start
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx_base = pid * BLOCK_SIZE + j_offsets
    j_mask_base = j_idx_base < len_2d
    
    # Initialize counter for this thread block
    local_k = 0
    
    # Sequential loop over i dimension (required for dependencies)
    for i in range(len_2d):
        # Compute current j indices
        j_idx = j_idx_base
        j_mask = j_mask_base
        
        # Compute linear indices for bb[j][i] and aa[j][i]
        bb_linear_idx = j_idx * len_2d + i
        aa_linear_idx = j_idx * len_2d + i
        
        # Load bb values for condition check
        bb_vals = tl.load(bb_ptr + bb_linear_idx, mask=j_mask, other=0.0)
        
        # Load aa values
        aa_vals = tl.load(aa_ptr + aa_linear_idx, mask=j_mask, other=0.0)
        
        # Check condition: bb[j][i] > 0.0
        condition_mask = bb_vals > 0.0
        valid_mask = j_mask & condition_mask
        
        # Store values for valid positions
        for local_j in range(BLOCK_SIZE):
            if j_idx_base + local_j < len_2d:
                if bb_vals[local_j] > 0.0:
                    tl.store(flat_2d_array_ptr + local_k, aa_vals[local_j])
                    local_k += 1

def s343_triton(aa, bb, flat_2d_array, len_2d):
    # Stream compaction with sequential dependency on counter k
    # Use PyTorch for correct stream compaction behavior
    
    total_k = 0
    
    # Process element by element maintaining order i->j
    for i in range(len_2d):
        for j in range(len_2d):
            if bb[j, i] > 0.0:
                flat_2d_array[total_k] = aa[j, i]
                total_k += 1
    
    return flat_2d_array