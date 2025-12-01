import torch
import triton
import triton.language as tl

@triton.jit
def s125_kernel(flat_2d_array_ptr, aa_ptr, bb_ptr, cc_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    aa_vals = tl.load(aa_ptr + idx, mask=mask)
    bb_vals = tl.load(bb_ptr + idx, mask=mask)
    cc_vals = tl.load(cc_ptr + idx, mask=mask)
    
    result = aa_vals + bb_vals * cc_vals
    
    tl.store(flat_2d_array_ptr + idx, result, mask=mask)

def s125_triton(flat_2d_array, aa, bb, cc):
    flat_2d_array = flat_2d_array.contiguous()
    aa = aa.contiguous().view(-1)
    bb = bb.contiguous().view(-1)
    cc = cc.contiguous().view(-1)
    
    n_elements = aa.numel()
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    s125_kernel[(grid_size,)](
        flat_2d_array,
        aa,
        bb,
        cc,
        n_elements,
        BLOCK_SIZE
    )
    
    return flat_2d_array