import triton
import triton.language as tl
import torch

@triton.jit
def s141_kernel(flat_2d_array_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    k = (i + 1) * ((i + 1) - 1) // 2 + (i + 1) - 1
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for j_start in range(i, LEN_2D, BLOCK_SIZE):
        j_indices = j_start + j_offsets
        mask = (j_indices >= i) & (j_indices < LEN_2D)
        
        valid_j = tl.where(mask, j_indices, 0)
        bb_indices = valid_j * LEN_2D + i
        bb_values = tl.load(bb_ptr + bb_indices, mask=mask, other=0.0)
        
        k_offsets = tl.zeros_like(j_offsets, dtype=tl.int32)
        for idx in range(BLOCK_SIZE):
            if j_start + idx >= i and j_start + idx < LEN_2D:
                k_offsets = tl.where(j_offsets == idx, k + tl.sum(tl.arange(0, idx + 1) + j_start), k_offsets)
        
        valid_k = tl.where(mask, k_offsets, 0)
        
        for idx in range(BLOCK_SIZE):
            if j_start + idx >= i and j_start + idx < LEN_2D:
                current_mask = mask & (j_offsets == idx)
                if tl.sum(current_mask.to(tl.int32)) > 0:
                    k_val = tl.sum(tl.where(current_mask, valid_k, 0))
                    bb_val = tl.sum(tl.where(current_mask, bb_values, 0.0))
                    old_val = tl.load(flat_2d_array_ptr + k_val)
                    tl.store(flat_2d_array_ptr + k_val, old_val + bb_val)
        
        k += tl.sum(tl.arange(j_start, min(j_start + BLOCK_SIZE, LEN_2D)) + 1)

def s141_triton(flat_2d_array, bb):
    LEN_2D = bb.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(LEN_2D, 1),)
    
    s141_kernel[grid](
        flat_2d_array, bb, 
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )