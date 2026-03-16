import triton
import triton.language as tl
import torch

@triton.jit
def fdtd_2d_kernel(_fict_ptr, ex_ptr, ey_ptr, hz_ptr, 
                   NX: tl.constexpr, NY: tl.constexpr, TMAX: tl.constexpr):
    
    # Pre-compute offsets once
    j_offsets = tl.arange(0, NY)
    i_offsets = tl.arange(0, NX)
    
    for t in range(TMAX):
        # Phase 1: Set boundary conditions for ey[0][j]
        ey_boundary_offsets = 0 * NY + j_offsets
        fict_val = tl.load(_fict_ptr + t)
        tl.store(ey_ptr + ey_boundary_offsets, fict_val)
        
        # Phase 2: Update ey[i][j] for i >= 1
        for i in range(1, NX):
            ey_offsets = i * NY + j_offsets
            hz_offsets = i * NY + j_offsets
            hz_prev_offsets = (i - 1) * NY + j_offsets
            
            ey_vals = tl.load(ey_ptr + ey_offsets)
            hz_vals = tl.load(hz_ptr + hz_offsets)
            hz_prev_vals = tl.load(hz_ptr + hz_prev_offsets)
            
            new_ey = ey_vals - 0.5 * (hz_vals - hz_prev_vals)
            tl.store(ey_ptr + ey_offsets, new_ey)
        
        # Phase 3: Update ex[i][j] for j >= 1
        for j in range(1, NY):
            ex_offsets = i_offsets * NY + j
            hz_offsets = i_offsets * NY + j
            hz_prev_offsets = i_offsets * NY + (j - 1)
            
            ex_vals = tl.load(ex_ptr + ex_offsets)
            hz_vals = tl.load(hz_ptr + hz_offsets)
            hz_prev_vals = tl.load(hz_ptr + hz_prev_offsets)
            
            new_ex = ex_vals - 0.5 * (hz_vals - hz_prev_vals)
            tl.store(ex_ptr + ex_offsets, new_ex)
        
        # Phase 4: Update hz[i][j] for i < NX-1, j < NY-1
        j_range_offsets = tl.arange(0, NY - 1)
        for i in range(NX - 1):
            hz_offsets = i * NY + j_range_offsets
            ex_offsets = i * NY + (j_range_offsets + 1)
            ex_curr_offsets = i * NY + j_range_offsets
            ey_next_offsets = (i + 1) * NY + j_range_offsets
            ey_curr_offsets = i * NY + j_range_offsets
            
            hz_vals = tl.load(hz_ptr + hz_offsets)
            ex_next_vals = tl.load(ex_ptr + ex_offsets)
            ex_curr_vals = tl.load(ex_ptr + ex_curr_offsets)
            ey_next_vals = tl.load(ey_ptr + ey_next_offsets)
            ey_curr_vals = tl.load(ey_ptr + ey_curr_offsets)
            
            new_hz = hz_vals - 0.7 * ((ex_next_vals - ex_curr_vals) + 
                                      (ey_next_vals - ey_curr_vals))
            tl.store(hz_ptr + hz_offsets, new_hz)

def fdtd_2d_triton(_fict_, ex, ey, hz, NX, NY, TMAX):
    grid = (1,)
    fdtd_2d_kernel[grid](_fict_, ex, ey, hz, NX, NY, TMAX)