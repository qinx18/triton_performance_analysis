import triton
import triton.language as tl
import torch

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i_val, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < i_val
    
    # Load bb[j][i] values
    bb_indices = j_offsets * 256 + i_val
    bb_vals = tl.load(bb_ptr + bb_indices, mask=j_mask, other=0.0)
    
    # Load a[i-j-1] values  
    a_indices = i_val - j_offsets - 1
    a_mask = j_mask & (a_indices >= 0)
    a_vals = tl.load(a_ptr + a_indices, mask=a_mask, other=0.0)
    
    # Compute products and sum
    products = bb_vals * a_vals
    result = tl.sum(products, axis=0)
    
    # Add to a[i]
    if tl.program_id(0) == 0:
        current_a_i = tl.load(a_ptr + i_val)
        tl.store(a_ptr + i_val, current_a_i + result)

def s118_triton(a, bb):
    N = bb.shape[0]
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    for i in range(1, N):
        s118_kernel[(1,)](
            a, bb, i, BLOCK_SIZE
        )