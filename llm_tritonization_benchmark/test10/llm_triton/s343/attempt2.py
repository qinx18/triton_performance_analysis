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
    i = indices // LEN_2D
    j = indices % LEN_2D
    
    # Access bb[j][i] and aa[j][i]
    bb_indices = j * LEN_2D + i
    aa_indices = j * LEN_2D + i
    
    # Load bb values
    bb_vals = tl.load(bb_ptr + bb_indices, mask=mask, other=0.0)
    
    # Check condition bb[j][i] > 0.0
    condition = bb_vals > 0.0
    
    # Load aa values where condition is true
    aa_vals = tl.load(aa_ptr + aa_indices, mask=mask & condition, other=0.0)
    
    # Count valid elements and get positions
    for i in range(BLOCK_SIZE):
        if mask[i] and condition[i]:
            # Atomic increment to get next position
            pos = tl.atomic_add(count_ptr, 1)
            tl.store(flat_2d_array_ptr + pos, aa_vals[i])

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