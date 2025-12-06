import torch
import triton
import triton.language as tl

@triton.jit
def s141_kernel(
    bb_ptr,
    flat_2d_array_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Calculate initial k value
    k = (i + 1) * (i + 1 - 1) // 2 + (i + 1) - 1
    
    # Process elements in blocks
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for j_start in range(i, LEN_2D, BLOCK_SIZE):
        j_vals = j_start + j_offsets
        j_mask = (j_vals < LEN_2D) & (j_vals >= i)
        
        if tl.sum(j_mask.to(tl.int32)) == 0:
            break
        
        # Load bb[j][i] values
        bb_indices = j_vals * LEN_2D + i
        bb_vals = tl.load(bb_ptr + bb_indices, mask=j_mask, other=0.0)
        
        # Calculate k indices for this block
        k_vals = k + tl.cumsum(j_vals - i + 1, axis=0) - (j_vals - i + 1)
        
        # Store to flat_2d_array
        for idx in range(BLOCK_SIZE):
            if j_start + idx < LEN_2D and j_start + idx >= i:
                j_val = j_start + idx
                k_idx = k + tl.sum((tl.arange(0, j_val - i + 1) + i + 1))
                current_val = tl.load(flat_2d_array_ptr + k_idx)
                bb_val = tl.load(bb_ptr + j_val * LEN_2D + i)
                tl.store(flat_2d_array_ptr + k_idx, current_val + bb_val)

@triton.jit
def s141_kernel_simple(
    bb_ptr,
    flat_2d_array_ptr,
    LEN_2D: tl.constexpr,
):
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Calculate initial k value: (i+1) * ((i+1) - 1) / 2 + (i+1) - 1
    k = (i + 1) * i // 2 + i
    
    # Sequential loop over j from i to LEN_2D-1
    for j_offset in range(LEN_2D - i):
        j = i + j_offset
        if j >= LEN_2D:
            break
            
        # Load bb[j][i]
        bb_val = tl.load(bb_ptr + j * LEN_2D + i)
        
        # Load current flat_2d_array[k]
        current_val = tl.load(flat_2d_array_ptr + k)
        
        # Update flat_2d_array[k]
        tl.store(flat_2d_array_ptr + k, current_val + bb_val)
        
        # Update k: k += j+1
        k = k + j + 1

def s141_triton(bb, flat_2d_array):
    LEN_2D = bb.shape[0]
    
    # Launch kernel with one thread per i
    grid = (LEN_2D,)
    
    s141_kernel_simple[grid](
        bb,
        flat_2d_array,
        LEN_2D=LEN_2D,
    )
    
    return flat_2d_array