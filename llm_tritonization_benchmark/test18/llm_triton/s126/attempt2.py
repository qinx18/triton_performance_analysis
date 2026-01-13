import triton
import triton.language as tl
import torch

@triton.jit
def s126_kernel(bb_ptr, cc_ptr, flat_2d_array_ptr, N):
    i = tl.program_id(0)
    
    if i >= N:
        return
    
    # j = 1 to N-1, sequential processing due to dependency
    for j in range(1, N):
        # bb[j][i] = bb[j-1][i] + flat_2d_array[k-1] * cc[j][i]
        # k = 1 + i * N + (j - 1) = 1 + i * N + j - 1 = i * N + j
        k = i * N + j
        
        # Load bb[j-1][i]
        bb_prev_offset = (j - 1) * N + i
        bb_prev = tl.load(bb_ptr + bb_prev_offset)
        
        # Load cc[j][i]
        cc_offset = j * N + i
        cc_val = tl.load(cc_ptr + cc_offset)
        
        # Load flat_2d_array[k-1]
        flat_offset = k - 1
        flat_val = tl.load(flat_2d_array_ptr + flat_offset)
        
        # Compute and store bb[j][i]
        result = bb_prev + flat_val * cc_val
        tl.store(bb_ptr + cc_offset, result)

def s126_triton(bb, cc, flat_2d_array):
    N = bb.shape[0]
    
    grid = (N,)
    
    s126_kernel[grid](bb, cc, flat_2d_array, N)
    
    return bb