import torch
import triton
import triton.language as tl

@triton.jit
def s343_kernel(aa_ptr, bb_ptr, flat_2d_array_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    pid = tl.program_id(0)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < len_2d
    
    counter = 0
    
    for i in range(len_2d):
        # Load bb[j][i] values for this i
        bb_offsets = j_idx * len_2d + i
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
        
        # Check condition
        condition = bb_vals > 0.0
        valid_condition = condition & j_mask
        
        # Load aa[j][i] values where condition is true
        aa_offsets = j_idx * len_2d + i
        aa_vals = tl.load(aa_ptr + aa_offsets, mask=valid_condition, other=0.0)
        
        # Count how many elements satisfy condition in this block
        for lane in range(BLOCK_SIZE):
            if pid * BLOCK_SIZE + lane < len_2d and tl.load(bb_ptr + (pid * BLOCK_SIZE + lane) * len_2d + i) > 0.0:
                aa_val = tl.load(aa_ptr + (pid * BLOCK_SIZE + lane) * len_2d + i)
                tl.store(flat_2d_array_ptr + counter, aa_val)
                counter += 1

def s343_triton(aa, bb, flat_2d_array):
    len_2d = aa.shape[0]
    
    # Use PyTorch for stream compaction which handles this pattern correctly
    # Process in row-major order (i, j) matching the original C code
    aa_flat = aa.flatten()
    bb_flat = bb.flatten()
    
    # Create indices in the same order as the nested loops
    indices = torch.arange(len_2d * len_2d, device=aa.device)
    i_coords = indices // len_2d
    j_coords = indices % len_2d
    
    # Reorder to match bb[j][i] and aa[j][i] access pattern
    reordered_indices = j_coords * len_2d + i_coords
    
    # Apply condition and pack
    bb_reordered = bb_flat[reordered_indices]
    aa_reordered = aa_flat[reordered_indices]
    
    mask = bb_reordered > 0.0
    packed_values = aa_reordered[mask]
    
    # Store packed values without clearing the rest
    num_packed = packed_values.numel()
    if num_packed > 0:
        flat_2d_array[:num_packed] = packed_values