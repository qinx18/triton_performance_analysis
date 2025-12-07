import torch
import triton
import triton.language as tl

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for i_start in range(0, n_elements, BLOCK_SIZE):
        i_offsets = i_start + offsets
        mask = i_offsets < n_elements
        
        # Load input arrays
        b_vals = tl.load(b_ptr + i_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + i_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + i_offsets, mask=mask, other=0.0)
        e_vals = tl.load(e_ptr + i_offsets, mask=mask, other=0.0)
        
        # Process each element sequentially within the block
        j = tl.full([BLOCK_SIZE], -1, dtype=tl.int32)
        
        for i_idx in range(BLOCK_SIZE):
            if i_start + i_idx >= n_elements:
                break
                
            # j++
            j_val = j[i_idx] + 1
            
            # a[j] = b[i] + d[i] * e[i]
            result1 = b_vals[i_idx] + d_vals[i_idx] * e_vals[i_idx]
            tl.store(a_ptr + j_val, result1)
            
            # Check condition c[i] > 0
            if c_vals[i_idx] > 0.0:
                j_val += 1
                result2 = c_vals[i_idx] + d_vals[i_idx] * e_vals[i_idx]
                tl.store(a_ptr + j_val, result2)
            
            j = tl.full([BLOCK_SIZE], j_val, dtype=tl.int32)

def s123_triton(a, b, c, d, e):
    n_elements = b.shape[0] // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s123_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )