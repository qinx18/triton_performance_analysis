import torch
import triton
import triton.language as tl

@triton.jit
def s141_kernel(bb_ptr, flat_2d_array_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Each program handles one value of i
    i = pid
    if i >= N:
        return
    
    # Calculate starting position k for this i
    # k = (i+1) * ((i+1) - 1) / 2 + (i+1) - 1
    # Simplifying: k = (i+1) * i / 2 + i = i * (i + 1) / 2 + i
    k_start = i * (i + 1) // 2 + i
    
    # Process j loop in blocks
    for j_block_start in range(i, N, BLOCK_SIZE):
        # Create offsets for this block
        j_offsets = tl.arange(0, BLOCK_SIZE)
        j_indices = j_block_start + j_offsets
        
        # Mask for valid j values (j >= i and j < N)
        valid_mask = (j_indices >= i) & (j_indices < N)
        
        if not tl.sum(valid_mask.to(tl.int32)):
            continue
        
        # Calculate k values for this block of j
        # k starts at k_start for j=i, then k += j+1 for each subsequent j
        # So for j in [i, i+1, i+2, ...], k values are:
        # k_i, k_i + (i+1), k_i + (i+1) + (i+2), ...
        
        # Calculate cumulative sum of (j+1) from i to current j
        k_offsets = tl.zeros_like(j_offsets, dtype=tl.int32)
        for idx in range(BLOCK_SIZE):
            if j_block_start + idx >= i and j_block_start + idx < N:
                # Calculate k offset for this j
                j_val = j_block_start + idx
                # k offset = sum of (j'+1) for j' from i to j_val-1
                k_offset = 0
                for j_prime in range(i, j_val):
                    k_offset += j_prime + 1
                k_offsets = tl.where(j_offsets == idx, k_start + k_offset, k_offsets)
        
        # Load bb[j][i] values
        bb_indices = j_indices * N + i
        bb_vals = tl.load(bb_ptr + bb_indices, mask=valid_mask, other=0.0)
        
        # Load current flat_2d_array values
        flat_vals = tl.load(flat_2d_array_ptr + k_offsets, mask=valid_mask, other=0.0)
        
        # Update flat_2d_array[k] += bb[j][i]
        new_vals = flat_vals + bb_vals
        tl.store(flat_2d_array_ptr + k_offsets, new_vals, mask=valid_mask)

def s141_triton(bb, flat_2d_array):
    N = bb.shape[0]
    
    BLOCK_SIZE = 64
    grid = (N,)
    
    s141_kernel[grid](
        bb, flat_2d_array, 
        N=N, BLOCK_SIZE=BLOCK_SIZE
    )