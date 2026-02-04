import triton
import triton.language as tl
import torch

@triton.jit
def s118_kernel(a_ptr, bb_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(1, len_2d):
        j_max = i - 1
        
        # Process j in blocks
        accumulator = 0.0
        for j_start in range(0, i, BLOCK_SIZE):
            j_indices = j_start + j_offsets
            j_mask = (j_indices <= j_max) & (j_indices >= 0)
            
            # Load bb[j][i]
            bb_indices = j_indices * len_2d + i
            bb_vals = tl.load(bb_ptr + bb_indices, mask=j_mask, other=0.0)
            
            # Load a[i-j-1]
            a_read_indices = i - j_indices - 1
            a_read_mask = j_mask & (a_read_indices >= 0) & (a_read_indices < len_2d)
            a_vals = tl.load(a_ptr + a_read_indices, mask=a_read_mask, other=0.0)
            
            # Compute products and accumulate
            products = bb_vals * a_vals
            masked_products = tl.where(j_mask, products, 0.0)
            accumulator += tl.sum(masked_products)
        
        # Load current a[i] and update
        current_a = tl.load(a_ptr + i)
        new_a = current_a + accumulator
        tl.store(a_ptr + i, new_a)

def s118_triton(a, bb, len_2d):
    BLOCK_SIZE = 256
    
    grid = (1,)
    s118_kernel[grid](a, bb, len_2d, BLOCK_SIZE)
    
    return a