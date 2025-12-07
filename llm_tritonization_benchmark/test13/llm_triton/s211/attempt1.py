import triton
import triton.language as tl
import torch

@triton.jit
def s211_kernel(
    a_ptr, b_ptr, b_copy_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get block offsets
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process prologue: a[1] uses original b[0]
    if tl.program_id(0) == 0:
        # Load original b[0] for a[1]
        b_0 = tl.load(b_copy_ptr)
        c_1 = tl.load(c_ptr + 1)
        d_1 = tl.load(d_ptr + 1)
        a_1 = b_0 + c_1 * d_1
        tl.store(a_ptr + 1, a_1)
    
    # Main loop: reordered computation
    for block_start in range(1, n_elements - 1, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < (n_elements - 1)
        
        # Step 1: Compute b[i] (producer)
        b_vals = tl.load(b_copy_ptr + current_offsets + 1, mask=mask)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        new_b = b_vals - e_vals * d_vals
        tl.store(b_ptr + current_offsets, new_b, mask=mask)
        
        # Step 2: Compute a[i+1] using b[i] (shifted consumer)
        next_offsets = current_offsets + 1
        next_mask = (next_offsets < n_elements) & mask
        if tl.any(next_mask):
            c_next = tl.load(c_ptr + next_offsets, mask=next_mask)
            d_next = tl.load(d_ptr + next_offsets, mask=next_mask)
            new_a = new_b + c_next * d_next
            tl.store(a_ptr + next_offsets, new_a, mask=next_mask)

def s211_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    # Create read-only copy to handle WAR dependency
    b_copy = b.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s211_kernel[grid](
        a, b, b_copy, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )