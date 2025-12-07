import torch
import triton
import triton.language as tl

@triton.jit
def s141_kernel(bb_ptr, flat_2d_array_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for outer i loop
    pid = tl.program_id(0)
    i = pid
    
    if i >= LEN_2D:
        return
    
    # Calculate initial k
    k = (i + 1) * i // 2 + i
    
    # Process inner j loop in blocks
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for j_start in range(i, LEN_2D, BLOCK_SIZE):
        current_j = j_start + j_offsets
        j_mask = (current_j < LEN_2D) & (current_j >= i)
        
        # Load bb[j][i] values
        bb_indices = current_j * LEN_2D + i
        bb_vals = tl.load(bb_ptr + bb_indices, mask=j_mask, other=0.0)
        
        # Calculate k values for each j
        k_vals = k + tl.cumsum(current_j, axis=0) - tl.cumsum(tl.full([BLOCK_SIZE], i, dtype=tl.int32), axis=0)
        
        # Adjust k_vals calculation
        k_base = k
        for idx in range(BLOCK_SIZE):
            if j_start + idx >= i and j_start + idx < LEN_2D:
                actual_j = j_start + idx
                actual_k = k_base + (actual_j - i) * (actual_j + i + 1) // 2
                
                # Load current value, add bb value, store back
                if actual_k < LEN_2D * LEN_2D:
                    current_val = tl.load(flat_2d_array_ptr + actual_k)
                    bb_val = tl.load(bb_ptr + actual_j * LEN_2D + i)
                    new_val = current_val + bb_val
                    tl.store(flat_2d_array_ptr + actual_k, new_val)

@triton.jit
def s141_kernel_simple(bb_ptr, flat_2d_array_ptr, LEN_2D: tl.constexpr):
    # Use single thread approach due to complex indexing
    pid = tl.program_id(0)
    i = pid
    
    if i >= LEN_2D:
        return
    
    # Calculate initial k: (i+1) * ((i+1) - 1) / 2 + (i+1) - 1 = i * (i+1) / 2 + i
    k = i * (i + 1) // 2 + i
    
    # Process j loop sequentially
    for j_offset in range(LEN_2D - i):
        j = i + j_offset
        if j >= LEN_2D:
            break
            
        # Load bb[j][i]
        bb_idx = j * LEN_2D + i
        bb_val = tl.load(bb_ptr + bb_idx)
        
        # Load current flat_2d_array[k], add bb_val, store back
        current_val = tl.load(flat_2d_array_ptr + k)
        new_val = current_val + bb_val
        tl.store(flat_2d_array_ptr + k, new_val)
        
        # Update k += j + 1
        k += j + 1

def s141_triton(bb, flat_2d_array):
    LEN_2D = bb.shape[0]
    
    # Launch kernel with one thread per outer i
    grid = (LEN_2D,)
    
    s141_kernel_simple[grid](
        bb, flat_2d_array, LEN_2D
    )
    
    return flat_2d_array