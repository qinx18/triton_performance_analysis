import triton
import triton.language as tl
import torch

@triton.jit
def s126_kernel(bb_ptr, cc_ptr, flat_2d_array_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i = pid
    
    if i >= N:
        return
    
    k = 1 + i * N + i
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process j loop in blocks
    for j_start in range(1, N, BLOCK_SIZE):
        j_indices = j_start + j_offsets
        mask = (j_indices < N) & (j_indices >= 1)
        
        # Load bb[j-1][i]
        prev_j_indices = j_indices - 1
        prev_offsets = prev_j_indices * N + i
        bb_prev = tl.load(bb_ptr + prev_offsets, mask=mask)
        
        # Load cc[j][i]
        cc_offsets = j_indices * N + i
        cc_vals = tl.load(cc_ptr + cc_offsets, mask=mask)
        
        # Load flat_2d_array values
        k_base = k + (j_start - 1)
        k_indices = k_base + j_offsets
        flat_offsets = k_indices - 1
        flat_mask = mask & (flat_offsets >= 0) & (flat_offsets < N * N)
        flat_vals = tl.load(flat_2d_array_ptr + flat_offsets, mask=flat_mask)
        
        # Compute bb[j][i]
        result = bb_prev + flat_vals * cc_vals
        
        # Store result
        tl.store(bb_ptr + cc_offsets, result, mask=mask)

def s126_triton(bb, cc, flat_2d_array):
    N = bb.shape[0]
    
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (N,)
    
    s126_kernel[grid](bb, cc, flat_2d_array, N, BLOCK_SIZE)
    
    return bb