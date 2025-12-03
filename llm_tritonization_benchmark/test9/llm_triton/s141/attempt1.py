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
    # Get program ID
    pid = tl.program_id(0)
    
    # Each program handles one value of i
    i = pid
    if i >= LEN_2D:
        return
    
    # Calculate initial k: k = (i+1) * ((i+1) - 1) / 2 + (i+1) - 1
    # Simplifying: k = i * (i + 1) / 2 + i = i * (i + 3) / 2
    k = i * (i + 1) // 2 + i
    
    # Process j from i to LEN_2D-1 in blocks
    j_start = i
    j_end = LEN_2D
    
    # Create offset vector once
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for j_block_start in range(j_start, j_end, BLOCK_SIZE):
        # Calculate current j values
        j_vals = j_block_start + offsets
        j_mask = j_vals < j_end
        
        # Load bb[j][i] values
        bb_offsets = j_vals * LEN_2D + i
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
        
        # For each j in this block, update flat_2d_array[k] and increment k
        for idx in range(BLOCK_SIZE):
            j = j_block_start + idx
            if j >= j_end:
                break
                
            # Calculate k for this specific j
            current_k = k + tl.sum(tl.arange(0, idx) + j_block_start - j_start + j_start + 1)
            
            # Load current value, add bb value, store back
            if idx == 0:
                # For first element, k is already correct
                actual_k = k
            else:
                # For subsequent elements, k increases by (j_prev + 1)
                actual_k = k
                for prev_idx in range(idx):
                    prev_j = j_block_start + prev_idx
                    actual_k += prev_j + 1
            
            old_val = tl.load(flat_2d_array_ptr + actual_k)
            new_val = old_val + tl.load(bb_ptr + j * LEN_2D + i)
            tl.store(flat_2d_array_ptr + actual_k, new_val)

@triton.jit  
def s141_kernel_serial(
    bb_ptr,
    flat_2d_array_ptr, 
    LEN_2D: tl.constexpr,
):
    # Serial implementation that matches C code exactly
    for i in range(LEN_2D):
        # k = (i+1) * ((i+1) - 1) / 2 + (i+1) - 1
        k = i * (i + 1) // 2 + i
        
        for j in range(i, LEN_2D):
            # flat_2d_array[k] += bb[j][i]
            old_val = tl.load(flat_2d_array_ptr + k)
            bb_val = tl.load(bb_ptr + j * LEN_2D + i)
            new_val = old_val + bb_val
            tl.store(flat_2d_array_ptr + k, new_val)
            
            # k += j + 1
            k += j + 1

def s141_triton(bb, flat_2d_array):
    LEN_2D = bb.shape[0]
    
    # Use serial kernel due to complex dependency pattern
    grid = (1,)
    s141_kernel_serial[grid](
        bb,
        flat_2d_array,
        LEN_2D=LEN_2D,
    )