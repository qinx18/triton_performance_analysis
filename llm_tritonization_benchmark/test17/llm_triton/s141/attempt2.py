import torch
import triton
import triton.language as tl

@triton.jit
def s141_kernel(
    bb_ptr,
    flat_2d_array_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid
    
    if i >= N:
        return
    
    # Calculate initial k: k = (i+1) * ((i+1) - 1) / 2 + (i+1) - 1
    # This simplifies to: k = i * (i + 1) / 2 + i = i * (i + 3) / 2
    k = i * (i + 3) // 2
    
    # Process the inner loop: for j in range(i, N)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for j_block_start in range(i, N, BLOCK_SIZE):
        current_j_offsets = j_block_start + j_offsets
        j_mask = (current_j_offsets < N) & (current_j_offsets >= i)
        
        # Check if any elements in this block are valid
        num_valid = tl.sum(j_mask.to(tl.int32))
        if num_valid == 0:
            break
            
        # Load bb[j][i] values
        bb_indices = current_j_offsets * N + i
        bb_vals = tl.load(bb_ptr + bb_indices, mask=j_mask, other=0.0)
        
        # Calculate corresponding k values for each j
        # k starts at i * (i + 3) / 2, then k += j+1 for each iteration
        # So k_j = k_initial + sum(j+1) for j from i to current_j-1
        # This gives: k_j = k_initial + (current_j - i) + sum(j) from i to current_j-1
        # sum(j) from i to current_j-1 = (current_j-1+i)*(current_j-i)/2
        k_base = k + (current_j_offsets - i) + (current_j_offsets - 1 + i) * (current_j_offsets - i) // 2
        
        # Load current flat_2d_array values
        flat_vals = tl.load(flat_2d_array_ptr + k_base, mask=j_mask, other=0.0)
        
        # Update flat_2d_array
        flat_vals = flat_vals + bb_vals
        tl.store(flat_2d_array_ptr + k_base, flat_vals, mask=j_mask)

def s141_triton(bb, flat_2d_array):
    N = bb.shape[0]
    
    BLOCK_SIZE = 64
    grid = (N,)
    
    s141_kernel[grid](
        bb,
        flat_2d_array,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )