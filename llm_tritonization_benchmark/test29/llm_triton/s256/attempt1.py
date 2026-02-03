import torch
import triton
import triton.language as tl

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    i_mask = i_offsets < N

    # Load a[0] once
    a0 = tl.load(a_ptr)

    for j in range(1, N):
        # Closed-form: a[j] = a0 if j even, else 1.0 - a0
        a_j = tl.where(j % 2 == 0, a0, 1.0 - a0)

        # Parallel load of bb[j][i] for all i
        bb_offsets = j * N + i_offsets
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=i_mask)
        d_j = tl.load(d_ptr + j)

        # Parallel compute and store aa[j][i]
        aa_vals = a_j + bb_vals * d_j
        aa_offsets = j * N + i_offsets
        tl.store(aa_ptr + aa_offsets, aa_vals, mask=i_mask)

        # Store a[j] (only first thread block needs to do this)
        if pid == 0:
            if i_offsets[0] == 0:
                tl.store(a_ptr + j, a_j)

def s256_triton(a, aa, bb, d, len_2d):
    N = len_2d
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s256_kernel[grid](
        a, aa, bb, d,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE
    )