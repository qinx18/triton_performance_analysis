import triton
import triton.language as tl
import torch

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_half, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    mask = i_offsets < n_half
    
    # Load input data
    b_vals = tl.load(b_ptr + i_offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + i_offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + i_offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + i_offsets, mask=mask, other=0.0)
    
    # First store: j++; a[j] = b[i] + d[i] * e[i];
    result1 = b_vals + d_vals * e_vals
    j_offsets = i_offsets
    tl.store(a_ptr + j_offsets, result1, mask=mask)
    
    # Conditional store: if (c[i] > 0) { j++; a[j] = c[i] + d[i] * e[i]; }
    c_positive = c_vals > 0.0
    result2 = c_vals + d_vals * e_vals
    
    # For the conditional j increment, we need to know how many c > 0 elements
    # exist before current position within this block
    c_int = tl.where(c_positive, 1, 0).to(tl.int32)
    
    # Calculate prefix sum manually for small block sizes
    prefix = c_int
    for shift in [1, 2, 4, 8, 16, 32, 64, 128]:
        if shift < BLOCK_SIZE:
            shifted_offsets = offsets - shift
            valid_shift_mask = shifted_offsets >= 0
            shifted_vals = tl.where(valid_shift_mask, tl.load(a_ptr + block_start + shifted_offsets, mask=valid_shift_mask, other=0), 0)
            # Use the current prefix values instead
            temp = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
            for k in range(BLOCK_SIZE):
                if k >= shift:
                    temp = tl.where(offsets == k, prefix + tl.where(offsets == (k - shift), prefix, 0), temp)
                else:
                    temp = tl.where(offsets == k, prefix, temp)
            prefix = tl.where(offsets >= shift, temp, prefix)
    
    # Calculate j for conditional store
    j_conditional = n_half + block_start + prefix - c_int + offsets
    
    # Store conditional result
    cond_store_mask = mask & c_positive
    tl.store(a_ptr + j_conditional, result2, mask=cond_store_mask)

def s123_triton(a, b, c, d, e):
    n = b.shape[0]
    n_half = n // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_half, BLOCK_SIZE),)
    
    s123_kernel[grid](
        a, b, c, d, e,
        n_half,
        BLOCK_SIZE=BLOCK_SIZE
    )