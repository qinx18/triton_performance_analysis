import triton
import triton.language as tl

@triton.jit
def s2233_kernel_aa(aa_ptr, cc_ptr, len_2d: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = pid * BLOCK_SIZE + i_offsets + 1
    i_mask = i_indices < len_2d
    
    for j in range(1, len_2d):
        j_prev = j - 1
        
        # Load aa[j-1, i]
        aa_prev_ptrs = aa_ptr + j_prev * len_2d + i_indices
        aa_prev = tl.load(aa_prev_ptrs, mask=i_mask, other=0.0)
        
        # Load cc[j, i]
        cc_ptrs = cc_ptr + j * len_2d + i_indices
        cc_vals = tl.load(cc_ptrs, mask=i_mask, other=0.0)
        
        # Compute aa[j, i] = aa[j-1, i] + cc[j, i]
        result = aa_prev + cc_vals
        
        # Store aa[j, i]
        aa_ptrs = aa_ptr + j * len_2d + i_indices
        tl.store(aa_ptrs, result, mask=i_mask)

@triton.jit
def s2233_kernel_bb(bb_ptr, cc_ptr, len_2d: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_indices = pid * BLOCK_SIZE + j_offsets + 1
    j_mask = j_indices < len_2d
    
    for i in range(1, len_2d):
        i_prev = i - 1
        
        # Load bb[i-1, j]
        bb_prev_ptrs = bb_ptr + i_prev * len_2d + j_indices
        bb_prev = tl.load(bb_prev_ptrs, mask=j_mask, other=0.0)
        
        # Load cc[i, j]
        cc_ptrs = cc_ptr + i * len_2d + j_indices
        cc_vals = tl.load(cc_ptrs, mask=j_mask, other=0.0)
        
        # Compute bb[i, j] = bb[i-1, j] + cc[i, j]
        result = bb_prev + cc_vals
        
        # Store bb[i, j]
        bb_ptrs = bb_ptr + i * len_2d + j_indices
        tl.store(bb_ptrs, result, mask=j_mask)

def s2233_triton(aa, bb, cc, len_2d):
    BLOCK_SIZE = 256
    
    # First kernel: parallelize across i, loop over j
    num_i_blocks = triton.cdiv(len_2d - 1, BLOCK_SIZE)
    grid = (num_i_blocks,)
    
    s2233_kernel_aa[grid](
        aa, cc, len_2d, BLOCK_SIZE
    )
    
    # Second kernel: parallelize across j, loop over i
    num_j_blocks = triton.cdiv(len_2d - 1, BLOCK_SIZE)
    grid = (num_j_blocks,)
    
    s2233_kernel_bb[grid](
        bb, cc, len_2d, BLOCK_SIZE
    )