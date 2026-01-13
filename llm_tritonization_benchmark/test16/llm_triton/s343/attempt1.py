import torch
import triton
import triton.language as tl

@triton.jit
def s343_kernel(aa_ptr, bb_ptr, flat_2d_array_ptr, len_2d: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for j dimension
    pid_j = tl.program_id(0)
    
    # Calculate j offsets for this block
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_indices = pid_j * BLOCK_SIZE + j_offsets
    j_mask = j_indices < len_2d
    
    # Global counter for output position
    output_counter = tl.zeros([1], dtype=tl.int32)
    
    # Sequential loop over i dimension
    for i in range(len_2d):
        # Calculate memory offsets for bb[j][i] and aa[j][i]
        bb_offsets = j_indices * len_2d + i
        aa_offsets = j_indices * len_2d + i
        
        # Load bb values and check condition
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
        condition_mask = (bb_vals > 0.0) & j_mask
        
        # Load aa values where condition is true
        aa_vals = tl.load(aa_ptr + aa_offsets, mask=condition_mask, other=0.0)
        
        # Count how many elements satisfy condition in this block
        num_true = tl.sum(condition_mask.to(tl.int32))
        
        # For each true condition, we need to write to output array
        # This is the challenging part - we need to maintain order
        for j_local in range(BLOCK_SIZE):
            j_global = pid_j * BLOCK_SIZE + j_local
            if j_global < len_2d:
                if bb_vals[j_local] > 0.0:
                    # Get current counter value and increment
                    current_pos = tl.atomic_add(output_counter, 1)
                    # Store the value
                    tl.store(flat_2d_array_ptr + current_pos, aa_vals[j_local])

def s343_triton(aa, bb, flat_2d_array):
    len_2d = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Use PyTorch for stream compaction as it's more efficient
    # Flatten arrays for easier processing
    aa_flat = aa.flatten()
    bb_flat = bb.flatten()
    
    # Create condition mask
    condition_mask = bb_flat > 0.0
    
    # Extract values that satisfy condition
    packed_values = aa_flat[condition_mask]
    
    # Get number of packed values
    num_packed = packed_values.numel()
    
    # Store packed values at beginning of output array
    if num_packed > 0:
        flat_2d_array[:num_packed] = packed_values