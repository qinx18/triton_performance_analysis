import torch
import triton
import triton.language as tl

def s343_triton(aa, bb, flat_2d_array):
    LEN_2D = aa.shape[0]
    
    # Use PyTorch for stream compaction pattern
    # Process in the correct order: i then j (column-major)
    aa_reordered = aa.t().flatten()  # Transpose then flatten to get [j][i] order
    bb_reordered = bb.t().flatten()  # Transpose then flatten to get [j][i] order
    
    mask = bb_reordered > 0.0
    packed_values = aa_reordered[mask]
    num_packed = packed_values.numel()
    
    if num_packed > 0:
        flat_2d_array[:num_packed] = packed_values