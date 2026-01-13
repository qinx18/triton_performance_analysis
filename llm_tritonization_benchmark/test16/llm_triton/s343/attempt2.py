import torch
import triton
import triton.language as tl

def s343_triton(aa, bb, flat_2d_array):
    len_2d = aa.shape[0]
    
    # Flatten arrays for processing
    aa_flat = aa.flatten()
    bb_flat = bb.flatten()
    
    # Process in row-major order (i outer loop, j inner loop)
    # Reshape to process in the correct order
    bb_ordered = bb_flat.view(len_2d, len_2d).t().flatten()  # transpose to get [j,i] order
    aa_ordered = aa_flat.view(len_2d, len_2d).t().flatten()  # transpose to get [j,i] order
    
    # Create condition mask
    condition_mask = bb_ordered > 0.0
    
    # Extract values that satisfy condition (stream compaction)
    packed_values = aa_ordered[condition_mask]
    
    # Get number of packed values
    num_packed = packed_values.numel()
    
    # Store packed values at beginning of output array
    if num_packed > 0:
        flat_2d_array[:num_packed] = packed_values