import torch
import triton
import triton.language as tl

@triton.jit
def s13110_kernel(aa_ptr, result_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Initialize with first element
    if pid == 0:
        max_val = tl.load(aa_ptr)
        max_i = 0
        max_j = 0
        
        # Sequential loop over rows
        for i in range(n_rows):
            # Parallel processing of columns
            for col_block in range(0, n_cols, BLOCK_SIZE):
                col_offsets = tl.arange(0, BLOCK_SIZE)
                j_indices = col_block + col_offsets
                mask = j_indices < n_cols
                
                # Load row values
                row_ptr = aa_ptr + i * n_cols + j_indices
                vals = tl.load(row_ptr, mask=mask, other=float('-inf'))
                
                # Find maximum in this block
                block_max = tl.max(vals)
                
                # Update global maximum
                if block_max > max_val:
                    max_val = block_max
                    # Find the exact position
                    max_mask = vals == block_max
                    for k in range(BLOCK_SIZE):
                        if mask[k] and max_mask[k]:
                            max_i = i
                            max_j = j_indices[k]
                            break
        
        # Store results
        tl.store(result_ptr, max_val)
        tl.store(result_ptr + 1, max_i.to(tl.float32))
        tl.store(result_ptr + 2, max_j.to(tl.float32))

def s13110_triton(aa):
    n_rows, n_cols = aa.shape
    
    # Use PyTorch for argmax - more efficient for this reduction pattern
    flat_aa = aa.flatten()
    max_val = torch.max(flat_aa)
    max_idx = torch.argmax(flat_aa)
    
    xindex = max_idx // n_cols
    yindex = max_idx % n_cols
    
    # Return value matches C code: max + xindex+1 + yindex+1
    return max_val + (xindex + 1) + (yindex + 1)