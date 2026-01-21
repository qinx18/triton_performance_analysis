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
    
    # Process j loop sequentially since k calculation depends on j
    k_current = k_start
    for j in range(i, N):
        # Load bb[j][i]
        bb_idx = j * N + i
        bb_val = tl.load(bb_ptr + bb_idx)
        
        # Load current flat_2d_array[k]
        flat_val = tl.load(flat_2d_array_ptr + k_current)
        
        # Update flat_2d_array[k] += bb[j][i]
        new_val = flat_val + bb_val
        tl.store(flat_2d_array_ptr + k_current, new_val)
        
        # Update k for next iteration: k += j+1
        k_current += j + 1

def s141_triton(bb, flat_2d_array):
    N = bb.shape[0]
    
    BLOCK_SIZE = 64
    grid = (N,)
    
    s141_kernel[grid](
        bb, flat_2d_array, 
        N=N, BLOCK_SIZE=BLOCK_SIZE
    )