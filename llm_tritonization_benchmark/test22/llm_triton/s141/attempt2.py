import torch
import triton
import triton.language as tl

@triton.jit
def s141_kernel(flat_2d_array_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # k = (i+1) * ((i+1) - 1) / 2 + (i+1)-1
    # Simplifying: k = i * (i + 1) / 2 + i = i * (i + 3) / 2
    k_base = i * (i + 1) // 2
    
    # Process j from i to LEN_2D-1 in blocks
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for j_start in range(i, LEN_2D, BLOCK_SIZE):
        j_end = min(j_start + BLOCK_SIZE, LEN_2D)
        
        if j_start >= LEN_2D:
            return
            
        j_vals = j_start + j_offsets
        j_mask = (j_vals >= i) & (j_vals < LEN_2D)
        
        # Load bb[j][i] values
        bb_offsets = j_vals * LEN_2D + i
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
        
        # Calculate k values for each j in the block
        # k = k_base + sum from i to j-1 of (m+1) where m goes from i to j-1
        # = k_base + sum from (i+1) to j of m
        # = k_base + j*(j+1)/2 - i*(i+1)/2
        j_extended = tl.where(j_mask, j_vals, i)
        k_vals = k_base + j_extended * (j_extended + 1) // 2 - i * (i + 1) // 2
        
        # Process each valid element
        for idx in range(BLOCK_SIZE):
            if j_start + idx >= LEN_2D:
                return
            if j_start + idx >= i:
                k_idx = k_base + (j_start + idx) * (j_start + idx + 1) // 2 - i * (i + 1) // 2
                bb_val = tl.load(bb_ptr + (j_start + idx) * LEN_2D + i)
                old_val = tl.load(flat_2d_array_ptr + k_idx)
                tl.store(flat_2d_array_ptr + k_idx, old_val + bb_val)

def s141_triton(bb, flat_2d_array):
    LEN_2D = bb.shape[0]
    
    BLOCK_SIZE = 64
    grid = (LEN_2D,)
    
    s141_kernel[grid](flat_2d_array, bb, LEN_2D, BLOCK_SIZE)