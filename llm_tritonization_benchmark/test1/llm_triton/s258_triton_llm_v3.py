import torch
import triton
import triton.language as tl

@triton.jit
def s258_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, aa_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Initialize s to 0
    s = 0.0
    
    # Process elements sequentially within each block
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        if idx < n_elements:
            # Load a[idx] and check condition
            a_val = tl.load(a_ptr + idx)
            if a_val > 0.0:
                d_val = tl.load(d_ptr + idx)
                s = d_val * d_val
            
            # Compute and store results
            c_val = tl.load(c_ptr + idx)
            d_val = tl.load(d_ptr + idx)
            aa_val = tl.load(aa_ptr + idx)  # aa[0][i]
            
            b_val = s * c_val + d_val
            e_val = (s + 1.0) * aa_val
            
            tl.store(b_ptr + idx, b_val)
            tl.store(e_ptr + idx, e_val)

def s258_triton(a, b, c, d, e, aa):
    n_elements = a.numel()
    
    # Use small block size to maintain sequential processing within blocks
    BLOCK_SIZE = 32
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # aa[0] corresponds to the first row of the 2D array
    aa_first_row = aa[0] if aa.dim() == 2 else aa.view(-1, int(aa.numel()**0.5))[0]
    
    s258_kernel[grid](
        a, b, c, d, e, aa_first_row,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )