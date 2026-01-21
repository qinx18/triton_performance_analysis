import triton
import triton.language as tl
import torch

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i, N, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    mask = j_offsets < i
    
    bb_offsets = j_offsets * N + i  # bb[j][i]
    a_read_offsets = i - j_offsets - 1  # a[i-j-1]
    
    # Ensure a_read_offsets are non-negative
    read_mask = mask & (a_read_offsets >= 0)
    
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask, other=0.0)
    a_vals = tl.load(a_ptr + a_read_offsets, mask=read_mask, other=0.0)
    
    products = bb_vals * a_vals
    sum_val = tl.sum(products, axis=0)
    
    # Update a[i]
    current_a_i = tl.load(a_ptr + i)
    new_a_i = current_a_i + sum_val
    tl.store(a_ptr + i, new_a_i)

def s118_triton(a, bb):
    N = bb.shape[0]  # LEN_2D
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    for i in range(1, N):
        s118_kernel[(1,)](a, bb, i, N, BLOCK_SIZE=BLOCK_SIZE)