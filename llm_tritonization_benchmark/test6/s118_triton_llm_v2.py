import torch
import triton
import triton.language as tl

@triton.jit
def s118_kernel(a_ptr, bb_ptr, N, i, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < i
    
    j_vals = offsets
    
    # Load bb[j, i] values
    bb_offsets = j_vals * N + i
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask, other=0.0)
    
    # Load a[i - j - 1] values
    a_read_indices = i - j_vals - 1
    a_read_mask = mask & (a_read_indices >= 0) & (a_read_indices < N)
    a_vals = tl.load(a_ptr + a_read_indices, mask=a_read_mask, other=0.0)
    
    # Compute products
    products = bb_vals * a_vals
    products = tl.where(mask, products, 0.0)
    
    # Sum reduction
    result = tl.sum(products)
    
    # Atomic add to a[i]
    if pid == 0:
        tl.atomic_add(a_ptr + i, result)
    elif tl.sum(mask) > 0:
        tl.atomic_add(a_ptr + i, result)

def s118_triton(a, bb):
    a = a.contiguous().clone()
    bb = bb.contiguous()
    
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential loop over i dimension
    for i in range(1, N):
        num_j = i
        if num_j > 0:
            grid = (triton.cdiv(num_j, BLOCK_SIZE),)
            s118_kernel[grid](a, bb, N, i, BLOCK_SIZE=BLOCK_SIZE)
    
    return a