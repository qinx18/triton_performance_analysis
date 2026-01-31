import triton
import triton.language as tl
import torch

@triton.jit
def s141_kernel(bb_ptr, flat_2d_array_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Each program handles one row i
    i = pid
    if i >= len_2d:
        return
    
    # Calculate initial k value: k = (i+1) * ((i+1) - 1) / 2 + (i+1) - 1
    # This simplifies to: k = i * (i + 1) / 2 + i = i * (i + 3) / 2
    k_init = i * (i + 1) // 2 + i
    
    # Process inner loop in blocks
    j_start = i
    j_end = len_2d
    
    block_offsets = tl.arange(0, BLOCK_SIZE)
    
    for j_block_start in range(j_start, j_end, BLOCK_SIZE):
        j_offsets = j_block_start + block_offsets
        mask = j_offsets < j_end
        
        # Calculate k values for this block
        # k starts at k_init + sum(j+1) for j from i to j_block_start-1
        # sum(j+1) from i to j_block_start-1 = sum from (i+1) to j_block_start
        # = j_block_start*(j_block_start+1)/2 - i*(i+1)/2
        if j_block_start > j_start:
            k_offset = (j_block_start - i) * (j_block_start + i + 1) // 2
        else:
            k_offset = 0
        k_base = k_init + k_offset
        
        # Load bb values: bb[j][i]
        bb_offsets = j_offsets * len_2d + i
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask, other=0.0)
        
        # Calculate k positions for each j in this block
        # k increments by (j+1) for each step
        for idx in range(BLOCK_SIZE):
            if j_block_start + idx >= j_end:
                break
            
            j_val = j_block_start + idx
            if j_val >= i:
                # Calculate k for this specific j
                k_for_j = k_base
                for temp_j in range(j_block_start, j_val):
                    if temp_j >= i:
                        k_for_j += temp_j + 1
                
                # Load current value and add bb value
                current_val = tl.load(flat_2d_array_ptr + k_for_j)
                bb_val = tl.load(bb_ptr + j_val * len_2d + i)
                new_val = current_val + bb_val
                tl.store(flat_2d_array_ptr + k_for_j, new_val)

def s141_triton(bb, flat_2d_array, len_2d):
    BLOCK_SIZE = 32
    grid = (len_2d,)
    
    s141_kernel[grid](
        bb, flat_2d_array, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )