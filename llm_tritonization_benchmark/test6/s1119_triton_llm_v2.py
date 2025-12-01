import torch
import triton
import triton.language as tl

@triton.jit
def s1119_kernel(
    aa_ptr, bb_ptr, N,
    BLOCK_SIZE: tl.constexpr,
):
    # Each block handles a contiguous chunk of j dimension
    block_id = tl.program_id(0)
    j_base = block_id * BLOCK_SIZE
    j_offsets = j_base + tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < N

    # In-kernel sequential loop over i
    for i in range(1, N):
        # Compute indices for current and previous i
        prev_offsets = (i - 1) * N + j_offsets
        curr_offsets = i * N + j_offsets

        # Load from previous i (same j column)
        prev_val = tl.load(aa_ptr + prev_offsets, mask=j_mask)
        bb_val = tl.load(bb_ptr + curr_offsets, mask=j_mask)

        # Compute and store
        result = prev_val + bb_val
        tl.store(aa_ptr + curr_offsets, result, mask=j_mask)

def s1119_triton(aa, bb):
    aa = aa.contiguous()
    bb = bb.contiguous()
    
    N = aa.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s1119_kernel[grid](
        aa, bb, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return aa