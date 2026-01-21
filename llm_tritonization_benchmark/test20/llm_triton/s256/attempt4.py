import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, j, i_offset, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = i_offset + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = i_offsets < LEN_2D
    
    # a[j] = 1.0 - a[j-1]
    a_prev = tl.load(a_ptr + j - 1)
    a_j = 1.0 - a_prev
    tl.store(a_ptr + j, a_j)
    
    # aa[j][i] = a[j] + bb[j][i]*d[j]
    bb_ptrs = bb_ptr + j * LEN_2D + i_offsets
    aa_ptrs = aa_ptr + j * LEN_2D + i_offsets
    d_j = tl.load(d_ptr + j)
    
    bb_vals = tl.load(bb_ptrs, mask=mask)
    aa_vals = a_j + bb_vals * d_j
    tl.store(aa_ptrs, aa_vals, mask=mask)

def s256_triton(a, aa, bb, d):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    for j in range(1, LEN_2D):
        for i_offset in range(0, LEN_2D, BLOCK_SIZE):
            grid = (1,)
            s256_kernel[grid](
                a, aa, bb, d, j, i_offset,
                LEN_2D=LEN_2D,
                BLOCK_SIZE=BLOCK_SIZE
            )