import torch
import triton
import triton.language as tl

@triton.jit
def s116_kernel(a_ptr, a_copy_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load from read-only copy
        a_vals = tl.load(a_copy_ptr + current_offsets, mask=mask)
        a_plus1 = tl.load(a_copy_ptr + current_offsets + 1, mask=mask & ((current_offsets + 1) < n_elements))
        a_plus2 = tl.load(a_copy_ptr + current_offsets + 2, mask=mask & ((current_offsets + 2) < n_elements))
        a_plus3 = tl.load(a_copy_ptr + current_offsets + 3, mask=mask & ((current_offsets + 3) < n_elements))
        a_plus4 = tl.load(a_copy_ptr + current_offsets + 4, mask=mask & ((current_offsets + 4) < n_elements))
        a_plus5 = tl.load(a_copy_ptr + current_offsets + 5, mask=mask & ((current_offsets + 5) < n_elements))
        
        # Compute results
        result_0 = a_plus1 * a_vals
        result_1 = a_plus2 * a_plus1
        result_2 = a_plus3 * a_plus2
        result_3 = a_plus4 * a_plus3
        result_4 = a_plus5 * a_plus4
        
        # Store to original array
        # For i positions
        store_mask_0 = mask & ((current_offsets % 5) == 0)
        tl.store(a_ptr + current_offsets, result_0, mask=store_mask_0)
        
        # For i+1 positions  
        store_mask_1 = mask & ((current_offsets % 5) == 1)
        tl.store(a_ptr + current_offsets, result_1, mask=store_mask_1)
        
        # For i+2 positions
        store_mask_2 = mask & ((current_offsets % 5) == 2)
        tl.store(a_ptr + current_offsets, result_2, mask=store_mask_2)
        
        # For i+3 positions
        store_mask_3 = mask & ((current_offsets % 5) == 3)
        tl.store(a_ptr + current_offsets, result_3, mask=store_mask_3)
        
        # For i+4 positions
        store_mask_4 = mask & ((current_offsets % 5) == 4)
        tl.store(a_ptr + current_offsets, result_4, mask=store_mask_4)

def s116_triton(a):
    n_elements = a.shape[0] - 5
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s116_kernel[grid](
        a,
        a_copy,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )