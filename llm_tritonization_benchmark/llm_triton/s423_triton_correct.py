import torch
import triton
import triton.language as tl

@triton.jit
def s423_kernel(
    flat_2d_array_ptr,
    a_ptr,
    n_elements,
    vl: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s423: flat_2d_array[i+1] = xx[i] + a[i]
    where xx = flat_2d_array + vl (pointer arithmetic)

    This means: flat_2d_array[i+1] = flat_2d_array[vl + i] + a[i]
    """
    # Get program ID and calculate element range for this block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask for valid elements to prevent out-of-bounds access
    mask = offsets < n_elements

    # Load input data with masking
    # xx[i] = flat_2d_array[vl + i]
    xx_offsets = vl + offsets
    xx_vals = tl.load(flat_2d_array_ptr + xx_offsets, mask=mask, other=0.0)
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)

    # Compute result: xx[i] + a[i]
    result = xx_vals + a_vals

    # Store to flat_2d_array[i+1] (offset by 1)
    output_offsets = offsets + 1
    tl.store(flat_2d_array_ptr + output_offsets, result, mask=mask)

def s423_triton(a, flat_2d_array):
    """
    Triton implementation of TSVC s423 function.

    Original C code:
    int vl = 64;
    xx = flat_2d_array + vl;
    for (int i = 0; i < LEN_1D - 1; i++) {
        flat_2d_array[i+1] = xx[i] + a[i];
    }

    This is equivalencing - xx is not a separate array but a pointer
    to flat_2d_array offset by 64 elements.
    """
    # Ensure tensors are contiguous for optimal memory access
    a = a.contiguous()
    flat_2d_array = flat_2d_array.contiguous()

    # Number of elements to process (excluding last element of a)
    n_elements = a.shape[0] - 1

    if n_elements <= 0:
        return flat_2d_array

    # The offset for xx pointer
    vl = 64

    # Choose block size for good occupancy
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)

    # Launch kernel with optimal grid configuration
    s423_kernel[(grid_size,)](
        flat_2d_array,
        a,
        n_elements,
        vl,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return flat_2d_array
