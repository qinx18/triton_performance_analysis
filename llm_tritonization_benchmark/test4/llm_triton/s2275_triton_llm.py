import torch
import triton
import triton.language as tl

@triton.jit
def s2275_kernel(
    a_ptr, aa_ptr, b_ptr, bb_ptr, c_ptr, cc_ptr, d_ptr,
    aa_out_ptr, a_out_ptr,
    n_elements_1d, n_elements_2d,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID for this block
    pid = tl.program_id(axis=0)
    
    # Calculate offsets for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Handle 2D array computation: aa = aa + bb * cc
    if block_start < n_elements_2d:
        mask_2d = offsets < n_elements_2d
        
        aa_vals = tl.load(aa_ptr + offsets, mask=mask_2d)
        bb_vals = tl.load(bb_ptr + offsets, mask=mask_2d)
        cc_vals = tl.load(cc_ptr + offsets, mask=mask_2d)
        
        aa_result = aa_vals + bb_vals * cc_vals
        tl.store(aa_out_ptr + offsets, aa_result, mask=mask_2d)
    
    # Handle 1D array computation: a = b + c * d
    if block_start < n_elements_1d:
        mask_1d = offsets < n_elements_1d
        
        b_vals = tl.load(b_ptr + offsets, mask=mask_1d)
        c_vals = tl.load(c_ptr + offsets, mask=mask_1d)
        d_vals = tl.load(d_ptr + offsets, mask=mask_1d)
        
        a_result = b_vals + c_vals * d_vals
        tl.store(a_out_ptr + offsets, a_result, mask=mask_1d)

def s2275_triton(a, aa, b, bb, c, cc, d):
    """
    Triton implementation of TSVC s2275.
    Optimized for parallel GPU execution with efficient memory access patterns.
    """
    # Ensure tensors are contiguous for optimal memory access
    a = a.contiguous()
    aa = aa.contiguous()
    b = b.contiguous()
    bb = bb.contiguous()
    c = c.contiguous()
    cc = cc.contiguous()
    d = d.contiguous()
    
    # Create output tensors
    a_out = torch.empty_like(a)
    aa_out = torch.empty_like(aa)
    
    # Get tensor dimensions
    n_elements_1d = a.numel()
    n_elements_2d = aa.numel()
    
    # Use larger block size for better GPU utilization
    BLOCK_SIZE = 1024
    
    # Calculate grid size to cover the maximum number of elements
    max_elements = max(n_elements_1d, n_elements_2d)
    grid_size = triton.cdiv(max_elements, BLOCK_SIZE)
    
    # Launch kernel with optimized block size
    s2275_kernel[grid_size,](
        a, aa, b, bb, c, cc, d,
        aa_out, a_out,
        n_elements_1d, n_elements_2d,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a_out, aa_out