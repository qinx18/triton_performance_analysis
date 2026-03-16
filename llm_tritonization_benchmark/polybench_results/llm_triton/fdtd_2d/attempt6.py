import triton
import triton.language as tl
import torch

@triton.jit
def fdtd_2d_kernel(_fict_ptr, ex_ptr, ey_ptr, hz_ptr, 
                   NX: tl.constexpr, NY: tl.constexpr, TMAX: tl.constexpr,
                   BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_start = pid_i * BLOCK_I
    j_start = pid_j * BLOCK_J
    
    i_offsets = i_start + tl.arange(0, BLOCK_I)
    j_offsets = j_start + tl.arange(0, BLOCK_J)
    
    for t in range(TMAX):
        # Step 1: ey[0][j] = _fict_[t]
        if pid_i == 0:
            fict_val = tl.load(_fict_ptr + t)
            j_mask = j_offsets < NY
            ey_idx = 0 * NY + j_offsets
            tl.store(ey_ptr + ey_idx, fict_val, mask=j_mask)
        
        tl.debug_barrier()
        
        # Step 2: ey[i][j] = ey[i][j] - 0.5*(hz[i][j]-hz[i-1][j]) for i >= 1
        if i_start >= 1:
            i_mask = i_offsets < NX
            j_mask = j_offsets < NY
            mask = i_mask[:, None] & j_mask[None, :]
            
            ey_idx = i_offsets[:, None] * NY + j_offsets[None, :]
            hz_idx = i_offsets[:, None] * NY + j_offsets[None, :]
            hz_prev_idx = (i_offsets[:, None] - 1) * NY + j_offsets[None, :]
            
            ey_vals = tl.load(ey_ptr + ey_idx, mask=mask)
            hz_vals = tl.load(hz_ptr + hz_idx, mask=mask)
            hz_prev_vals = tl.load(hz_ptr + hz_prev_idx, mask=mask)
            
            new_ey = ey_vals - 0.5 * (hz_vals - hz_prev_vals)
            tl.store(ey_ptr + ey_idx, new_ey, mask=mask)
        
        tl.debug_barrier()
        
        # Step 3: ex[i][j] = ex[i][j] - 0.5*(hz[i][j]-hz[i][j-1]) for j >= 1
        if j_start >= 1:
            i_mask = i_offsets < NX
            j_mask = j_offsets < NY
            mask = i_mask[:, None] & j_mask[None, :]
            
            ex_idx = i_offsets[:, None] * NY + j_offsets[None, :]
            hz_idx = i_offsets[:, None] * NY + j_offsets[None, :]
            hz_prev_idx = i_offsets[:, None] * NY + (j_offsets[None, :] - 1)
            
            ex_vals = tl.load(ex_ptr + ex_idx, mask=mask)
            hz_vals = tl.load(hz_ptr + hz_idx, mask=mask)
            hz_prev_vals = tl.load(hz_ptr + hz_prev_idx, mask=mask)
            
            new_ex = ex_vals - 0.5 * (hz_vals - hz_prev_vals)
            tl.store(ex_ptr + ex_idx, new_ex, mask=mask)
        
        tl.debug_barrier()
        
        # Step 4: hz[i][j] = hz[i][j] - 0.7*(ex[i][j+1] - ex[i][j] + ey[i+1][j] - ey[i][j])
        i_mask = i_offsets < (NX - 1)
        j_mask = j_offsets < (NY - 1)
        mask = i_mask[:, None] & j_mask[None, :]
        
        hz_idx = i_offsets[:, None] * NY + j_offsets[None, :]
        ex_idx = i_offsets[:, None] * NY + j_offsets[None, :]
        ex_next_idx = i_offsets[:, None] * NY + (j_offsets[None, :] + 1)
        ey_idx = i_offsets[:, None] * NY + j_offsets[None, :]
        ey_next_idx = (i_offsets[:, None] + 1) * NY + j_offsets[None, :]
        
        hz_vals = tl.load(hz_ptr + hz_idx, mask=mask)
        ex_vals = tl.load(ex_ptr + ex_idx, mask=mask)
        ex_next_vals = tl.load(ex_ptr + ex_next_idx, mask=mask)
        ey_vals = tl.load(ey_ptr + ey_idx, mask=mask)
        ey_next_vals = tl.load(ey_ptr + ey_next_idx, mask=mask)
        
        new_hz = hz_vals - 0.7 * (ex_next_vals - ex_vals + ey_next_vals - ey_vals)
        tl.store(hz_ptr + hz_idx, new_hz, mask=mask)
        
        tl.debug_barrier()

def fdtd_2d_triton(_fict_, ex, ey, hz, NX, NY, TMAX):
    BLOCK_I = 16
    BLOCK_J = 16
    
    grid_i = triton.cdiv(NX, BLOCK_I)
    grid_j = triton.cdiv(NY, BLOCK_J)
    
    fdtd_2d_kernel[(grid_i, grid_j)](
        _fict_, ex, ey, hz,
        NX, NY, TMAX,
        BLOCK_I, BLOCK_J
    )