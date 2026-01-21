import triton
import triton.language as tl

@triton.jit
def s241_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load values for first computation
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # First computation: a[i] = b[i] * c[i] * d[i]
        new_a_vals = b_vals * c_vals * d_vals
        tl.store(a_ptr + current_offsets, new_a_vals, mask=mask)
        
        # Load a[i+1] from the original copy for second computation
        next_offsets = current_offsets + 1
        next_mask = next_offsets < (a_copy_ptr.shape[0])
        a_copy_next_vals = tl.load(a_copy_ptr + next_offsets, mask=mask & next_mask, other=0.0)
        
        # Second computation: b[i] = a[i] * a[i+1] * d[i]
        new_b_vals = new_a_vals * a_copy_next_vals * d_vals
        tl.store(b_ptr + current_offsets, new_b_vals, mask=mask)

def s241_triton(a, b, c, d):
    n = a.shape[0] - 1  # Loop goes to LEN_1D-1
    
    # Create read-only copy to handle WAR dependency
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s241_kernel[grid](a, a_copy, b, c, d, n, BLOCK_SIZE=BLOCK_SIZE)