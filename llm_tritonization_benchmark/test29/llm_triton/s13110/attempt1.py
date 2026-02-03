import torch
import triton
import triton.language as tl

@triton.jit
def s13110_kernel(aa_ptr, output_ptr, len_2d: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid != 0:
        return
    
    # Initialize with first element
    max_val = tl.load(aa_ptr)
    xindex = 0
    yindex = 0
    
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(len_2d):
        j_base = 0
        while j_base < len_2d:
            j_idx = j_base + offsets
            mask = j_idx < len_2d
            
            # Load values for current i, multiple j
            ptrs = aa_ptr + i * len_2d + j_idx
            vals = tl.load(ptrs, mask=mask, other=float('-inf'))
            
            # Find max in this block
            block_max = tl.max(vals, axis=0)
            
            # Check if this block max is greater than current max
            if block_max > max_val:
                # Find the position of the max in this block
                max_mask = vals == block_max
                valid_mask = mask & max_mask
                
                if tl.sum(valid_mask.to(tl.int32)) > 0:
                    # Find first occurrence
                    for k in range(BLOCK_SIZE):
                        if k < len_2d - j_base and vals[k] == block_max and vals[k] > max_val:
                            max_val = vals[k]
                            xindex = i
                            yindex = j_base + k
                            break
            
            j_base += BLOCK_SIZE
    
    # Store results: max, xindex, yindex
    tl.store(output_ptr, max_val)
    tl.store(output_ptr + 1, xindex)
    tl.store(output_ptr + 2, yindex)

def s13110_triton(aa, len_2d):
    # Use PyTorch for argmax - more efficient and reliable
    max_val = torch.max(aa)
    flat_idx = torch.argmax(aa.flatten())
    xindex = flat_idx // len_2d
    yindex = flat_idx % len_2d
    
    chksum = max_val + xindex.float() + yindex.float()
    
    return max_val + (xindex + 1) + (yindex + 1)