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
    for j_start in range(i, LEN_2D, BLOCK_SIZE):
        j_end = min(j_start + BLOCK_SIZE, LEN_2D)
        block_size = j_end - j_start
        
        if block_size <= 0:
            break
            
        j_offsets = tl.arange(0, BLOCK_SIZE)
        j_vals = j_start + j_offsets
        j_mask = (j_vals >= i) & (j_vals < LEN_2D)
        
        # Load bb[j][i] values
        bb_offsets = j_vals * LEN_2D + i
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
        
        # Calculate k values for this block
        # k starts at k_base + sum from i to j_start-1 of (j+1)
        # sum from i to j_start-1 of (j+1) = sum from (i+1) to j_start of j
        # = j_start*(j_start+1)/2 - i*(i+1)/2
        if j_start > i:
            k_offset = j_start * (j_start + 1) // 2 - i * (i + 1) // 2
        else:
            k_offset = 0
        
        k_start = k_base + k_offset
        
        # For each j in the block, k += j+1 after the operation
        # So k values are: k_start, k_start + (j_start+1), k_start + (j_start+1) + (j_start+2), ...
        k_vals = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
        for idx in range(BLOCK_SIZE):
            if idx == 0:
                k_vals = tl.where(j_offsets == 0, k_start, k_vals)
            else:
                prev_k = tl.where(j_offsets == idx-1, k_vals, 0)
                prev_k_sum = tl.sum(prev_k)
                new_k = prev_k_sum + j_start + idx
                k_vals = tl.where(j_offsets == idx, new_k, k_vals)
        
        # Actually, let's compute k values directly
        # k = k_base + sum from i to j-1 of (m+1) where m goes from i to j-1
        # = k_base + sum from (i+1) to j of m
        # = k_base + j*(j+1)/2 - i*(i+1)/2
        j_extended = tl.where(j_mask, j_vals, i)  # Use i as default to avoid out of bounds
        k_vals = k_base + j_extended * (j_extended + 1) // 2 - i * (i + 1) // 2
        
        # Update flat_2d_array[k] += bb[j][i]
        for idx in range(BLOCK_SIZE):
            if j_start + idx < LEN_2D and j_start + idx >= i:
                j_val = j_start + idx
                k_val = k_base + j_val * (j_val + 1) // 2 - i * (i + 1) // 2
                bb_val = bb_vals[idx] if idx < block_size else 0.0
                if j_mask[idx]:
                    old_val = tl.load(flat_2d_array_ptr + k_val)
                    tl.store(flat_2d_array_ptr + k_val, old_val + bb_val)

def s141_triton(bb, flat_2d_array):
    LEN_2D = bb.shape[0]
    
    BLOCK_SIZE = 64
    grid = (LEN_2D,)
    
    s141_kernel[grid](flat_2d_array, bb, LEN_2D, BLOCK_SIZE)