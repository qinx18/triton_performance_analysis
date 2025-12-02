import torch
import triton
import triton.language as tl

@triton.jit
def s235_kernel(a_ptr, aa_ptr, b_ptr, bb_ptr, c_ptr, 
                n_elements, stride_aa_j, stride_aa_i,
                stride_bb_j, stride_bb_i,
                BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    mask = i_offsets < n_elements
    
    # Load b[i] and c[i]
    b_vals = tl.load(b_ptr + i_offsets, mask=mask)
    c_vals = tl.load(c_ptr + i_offsets, mask=mask)
    
    # Load a[i]
    a_vals = tl.load(a_ptr + i_offsets, mask=mask)
    
    # Update a[i] += b[i] * c[i]
    new_a_vals = a_vals + b_vals * c_vals
    tl.store(a_ptr + i_offsets, new_a_vals, mask=mask)
    
    # Inner loop: for j = 1 to n_elements-1
    for j in range(1, n_elements):
        # aa[j][i] = aa[j-1][i] + bb[j][i] * a[i]
        aa_prev_ptrs = aa_ptr + (j-1) * stride_aa_j + i_offsets * stride_aa_i
        aa_curr_ptrs = aa_ptr + j * stride_aa_j + i_offsets * stride_aa_i
        bb_ptrs = bb_ptr + j * stride_bb_j + i_offsets * stride_bb_i
        
        aa_prev_vals = tl.load(aa_prev_ptrs, mask=mask)
        bb_vals = tl.load(bb_ptrs, mask=mask)
        
        new_aa_vals = aa_prev_vals + bb_vals * new_a_vals
        tl.store(aa_curr_ptrs, new_aa_vals, mask=mask)

def s235_triton(a, aa, b, bb, c):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    stride_aa_j = aa.stride(0)
    stride_aa_i = aa.stride(1)
    stride_bb_j = bb.stride(0)
    stride_bb_i = bb.stride(1)
    
    s235_kernel[grid](
        a, aa, b, bb, c,
        n_elements,
        stride_aa_j, stride_aa_i,
        stride_bb_j, stride_bb_i,
        BLOCK_SIZE=BLOCK_SIZE
    )