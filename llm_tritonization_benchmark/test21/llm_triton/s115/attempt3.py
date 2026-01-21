import torch
import triton
import triton.language as tl

@triton.jit
def s115_kernel(a_ptr, aa_ptr, j, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    
    mask = (i_offsets > j) & (i_offsets < N)
    
    a_j = tl.load(a_ptr + j)
    
    aa_offsets = j * N + i_offsets
    aa_ji = tl.load(aa_ptr + aa_offsets, mask=mask)
    
    a_i = tl.load(a_ptr + i_offsets, mask=mask)
    
    result = a_i - aa_ji * a_j
    
    tl.store(a_ptr + i_offsets, result, mask=mask)

def s115_triton(a, aa):
    N = aa.shape[0]
    
    BLOCK_SIZE = 256
    
    for j in range(N):
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        s115_kernel[grid](a, aa, j, N, BLOCK_SIZE=BLOCK_SIZE)