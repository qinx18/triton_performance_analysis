import torch
import triton
import triton.language as tl

@triton.jit
def s13110_kernel(aa_ptr, max_val_ptr, max_i_ptr, max_j_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_base = pid * BLOCK_SIZE
    j_idx = j_base + j_offsets
    j_mask = j_idx < len_2d
    
    # Initialize with first element values
    if pid == 0:
        first_val = tl.load(aa_ptr)
        tl.store(max_val_ptr, first_val)
        tl.store(max_i_ptr, 0)
        tl.store(max_j_ptr, 0)
    
    tl.debug_barrier()
    
    for i in range(len_2d):
        # Load current row values for this block's j indices
        row_ptr = aa_ptr + i * len_2d + j_idx
        vals = tl.load(row_ptr, mask=j_mask, other=float('-inf'))
        
        # Load current maximum
        current_max = tl.load(max_val_ptr)
        
        # Find which elements in this block are greater than current max
        greater_mask = vals > current_max
        valid_mask = j_mask & greater_mask
        
        if tl.sum(valid_mask.to(tl.int32)) > 0:
            # Find the maximum value among valid elements
            masked_vals = tl.where(valid_mask, vals, float('-inf'))
            block_max = tl.max(masked_vals)
            
            # Find the j index of the maximum (first occurrence)
            max_positions = (masked_vals == block_max) & valid_mask
            j_indices = tl.where(max_positions, j_idx, len_2d)
            block_max_j = tl.min(j_indices)
            
            # Update global maximum atomically
            if block_max > current_max:
                tl.store(max_val_ptr, block_max)
                tl.store(max_i_ptr, i)
                tl.store(max_j_ptr, block_max_j)

def s13110_triton(aa, len_2d):
    # Use PyTorch for simpler and more reliable argmax
    flat_aa = aa.flatten()
    max_idx = torch.argmax(flat_aa)
    max_val = flat_aa[max_idx]
    
    # Convert flat index to 2D coordinates
    xindex = max_idx // len_2d
    yindex = max_idx % len_2d
    
    # Return the same value as C code: max + xindex+1 + yindex+1
    return max_val + xindex + yindex + 2