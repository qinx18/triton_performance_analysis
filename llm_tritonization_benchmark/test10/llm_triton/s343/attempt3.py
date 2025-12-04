import torch
import triton
import triton.language as tl

@triton.jit
def s343_kernel(aa_ptr, bb_ptr, flat_2d_array_ptr, count_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < LEN_2D * LEN_2D
    
    # Convert linear index to (i, j) coordinates following C code order
    i_coords = indices // LEN_2D
    j_coords = indices % LEN_2D
    
    # Access bb[j][i] and aa[j][i]
    bb_indices = j_coords * LEN_2D + i_coords
    aa_indices = j_coords * LEN_2D + i_coords
    
    # Load bb values
    bb_vals = tl.load(bb_ptr + bb_indices, mask=mask, other=0.0)
    
    # Check condition bb[j][i] > 0.0
    condition = bb_vals > 0.0
    final_mask = mask & condition
    
    # Load aa values where condition is true
    aa_vals = tl.load(aa_ptr + aa_indices, mask=final_mask, other=0.0)
    
    # Store values atomically
    element_offsets = tl.arange(0, BLOCK_SIZE)
    for elem_idx in range(BLOCK_SIZE):
        if elem_idx < BLOCK_SIZE:
            single_mask = (element_offsets == elem_idx) & final_mask
            if tl.sum(single_mask.to(tl.int32)) > 0:
                single_val = tl.sum(tl.where(single_mask, aa_vals, 0.0))
                pos = tl.atomic_add(count_ptr, 1)
                tl.store(flat_2d_array_ptr + pos, single_val)

def s343_triton(aa, bb, flat_2d_array):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Reset flat_2d_array
    flat_2d_array.zero_()
    
    # Counter for atomic operations
    count_tensor = torch.zeros(1, dtype=torch.int32, device=aa.device)
    
    grid = (triton.cdiv(LEN_2D * LEN_2D, BLOCK_SIZE),)
    
    s343_kernel[grid](
        aa, bb, flat_2d_array, count_tensor,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return flat_2d_array