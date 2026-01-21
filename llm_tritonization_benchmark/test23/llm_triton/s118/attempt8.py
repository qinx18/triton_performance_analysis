import torch
import triton
import triton.language as tl

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i_val, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate j range for this i
    j_start = pid * BLOCK_SIZE
    j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < i_val
    
    # Load bb[j][i] values - bb is stored in row-major order
    bb_indices = j_offsets * N + i_val
    bb_vals = tl.load(bb_ptr + bb_indices, mask=j_mask, other=0.0)
    
    # Load a[i-j-1] values
    a_read_indices = i_val - j_offsets - 1
    a_read_mask = j_mask & (a_read_indices >= 0) & (a_read_indices < N)
    a_vals = tl.load(a_ptr + a_read_indices, mask=a_read_mask, other=0.0)
    
    # Compute products and sum
    products = tl.where(a_read_mask, bb_vals * a_vals, 0.0)
    result = tl.sum(products)
    
    # Only first thread writes result for this i
    if pid == 0:
        tl.atomic_add(a_ptr + i_val, result)

def s118_triton(a, bb):
    N = bb.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential loop over i
    for i in range(1, N):
        num_j = i
        grid = (triton.cdiv(num_j, BLOCK_SIZE),)
        s118_kernel[grid](a, bb, i, N, BLOCK_SIZE)
    
    return a