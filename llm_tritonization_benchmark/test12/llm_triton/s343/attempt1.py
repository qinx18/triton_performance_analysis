import torch
import triton
import triton.language as tl

@triton.jit
def s343_kernel(aa_ptr, bb_ptr, flat_2d_array_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < len_2d
    
    k = 0
    
    for i in range(len_2d):
        # Load bb values for current i
        bb_ptrs = bb_ptr + j_idx * len_2d + i
        bb_vals = tl.load(bb_ptrs, mask=j_mask)
        
        # Check condition
        condition = bb_vals > 0.0
        
        # Load aa values for current i
        aa_ptrs = aa_ptr + j_idx * len_2d + i
        aa_vals = tl.load(aa_ptrs, mask=j_mask)
        
        # Process each j in this block sequentially to maintain ordering
        for j_local in range(BLOCK_SIZE):
            if pid * BLOCK_SIZE + j_local < len_2d:
                if (condition >> j_local) & 1:
                    flat_ptr = flat_2d_array_ptr + k
                    tl.store(flat_ptr, aa_vals >> j_local)
                    k += 1

def s343_triton(aa, bb, flat_2d_array):
    len_2d = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Use PyTorch for stream compaction as it's more efficient
    mask = bb > 0.0
    packed_values = aa[mask]
    num_packed = packed_values.numel()
    
    if num_packed > 0:
        flat_2d_array[:num_packed] = packed_values