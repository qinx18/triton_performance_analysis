import torch
import triton
import triton.language as tl

@triton.jit
def s1232_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate which triangle element this program handles
    total_elements = LEN_2D * (LEN_2D + 1) // 2
    
    if pid * BLOCK_SIZE >= total_elements:
        return
    
    # Pre-define offsets once
    offsets = tl.arange(0, BLOCK_SIZE)
    element_ids = pid * BLOCK_SIZE + offsets
    mask = element_ids < total_elements
    
    # Convert linear index to (i, j) coordinates for upper triangle
    # For upper triangle where i >= j, we need to find i, j from linear index
    i_coords = tl.zeros_like(element_ids)
    j_coords = tl.zeros_like(element_ids)
    
    # Calculate coordinates for valid elements
    for idx in range(BLOCK_SIZE):
        if pid * BLOCK_SIZE + idx < total_elements:
            elem_id = pid * BLOCK_SIZE + idx
            
            # Find j using quadratic formula: elem_id = j*LEN_2D - j*(j-1)/2 + (i-j)
            # Rearranging: j^2 - j*(2*LEN_2D + 1) + 2*elem_id = 0
            # But simpler: iterate to find j such that triangular number fits
            j = 0
            remaining = elem_id
            while j < LEN_2D:
                row_size = LEN_2D - j
                if remaining < row_size:
                    break
                remaining -= row_size
                j += 1
            
            i = j + remaining
            
            if idx < BLOCK_SIZE:
                i_coords = tl.where(offsets == idx, i, i_coords)
                j_coords = tl.where(offsets == idx, j, j_coords)
    
    # Calculate flat indices
    flat_indices = i_coords * LEN_2D + j_coords
    
    # Load values
    bb_vals = tl.load(bb_ptr + flat_indices, mask=mask, other=0.0)
    cc_vals = tl.load(cc_ptr + flat_indices, mask=mask, other=0.0)
    
    # Perform computation
    result = bb_vals + cc_vals
    
    # Store result
    tl.store(aa_ptr + flat_indices, result, mask=mask)

def s1232_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    
    # Total number of elements in upper triangle (i >= j)
    total_elements = LEN_2D * (LEN_2D + 1) // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    s1232_kernel[grid](
        aa, bb, cc,
        LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )