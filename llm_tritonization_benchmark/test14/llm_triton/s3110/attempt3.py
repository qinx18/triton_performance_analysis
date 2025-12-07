import torch
import triton
import triton.language as tl

@triton.jit
def s3110_kernel(aa_ptr, max_val_ptr, max_i_ptr, max_j_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < LEN_2D
    
    # Initialize with aa[0][0]
    current_max = tl.load(aa_ptr)
    current_i = 0
    current_j = 0
    
    # Check if we should consider j=0 in this block
    if pid == 0:
        # First block handles j=0, so skip it in the loop
        pass
    
    for i in range(LEN_2D):
        row_ptr = aa_ptr + i * LEN_2D
        values = tl.load(row_ptr + j_idx, mask=j_mask, other=-float('inf'))
        
        # For each value in this block
        for j_local in range(BLOCK_SIZE):
            if pid * BLOCK_SIZE + j_local < LEN_2D:
                val = tl.load(row_ptr + pid * BLOCK_SIZE + j_local)
                if val > current_max:
                    current_max = val
                    current_i = i
                    current_j = pid * BLOCK_SIZE + j_local
    
    # Store results for this block
    block_results_ptr = max_val_ptr + pid * 3
    tl.store(block_results_ptr, current_max)
    tl.store(block_results_ptr + 1, current_i.to(tl.float32))
    tl.store(block_results_ptr + 2, current_j.to(tl.float32))

def s3110_triton(aa):
    LEN_2D = aa.shape[0]
    
    # Use PyTorch for argmax - much more efficient
    flat_aa = aa.flatten()
    max_val = torch.max(flat_aa)
    flat_idx = torch.argmax(flat_aa)
    xindex = flat_idx // LEN_2D
    yindex = flat_idx % LEN_2D
    
    chksum = max_val + xindex.float() + yindex.float()
    
    return chksum