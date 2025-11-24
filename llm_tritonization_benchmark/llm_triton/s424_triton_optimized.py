import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel_strip(
    flat_2d_array_ptr,
    a_ptr,
    n_elements,
    vl: tl.constexpr,
    STRIP_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Single kernel that loops through all strips.
    Each thread processes one element per strip across all strips.
    """
    # Each thread gets a unique element ID within a strip (0 to STRIP_SIZE-1)
    tid = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Calculate number of strips
    num_strips = (n_elements + STRIP_SIZE - 1) // STRIP_SIZE

    # Loop through all strips (INSIDE the kernel)
    for strip_idx in range(num_strips):
        # Calculate actual element index for this thread in this strip
        element_idx = strip_idx * STRIP_SIZE + tid

        # Mask for valid elements
        mask = (tid < STRIP_SIZE) & (element_idx < n_elements)

        # Load input data: flat_2d_array[element_idx] + a[element_idx]
        flat_vals = tl.load(flat_2d_array_ptr + element_idx, mask=mask, other=0.0)
        a_vals = tl.load(a_ptr + element_idx, mask=mask, other=0.0)

        # Compute result
        result = flat_vals + a_vals

        # Store to flat_2d_array[vl + element_idx + 1]
        output_idx = vl + element_idx + 1
        tl.store(flat_2d_array_ptr + output_idx, result, mask=mask)

def s424_triton(a, flat_2d_array):
    """
    Optimized Triton implementation - Single kernel with loop inside.

    Launches ONE kernel where each thread loops through all strips.
    This eliminates kernel launch overhead from Python.
    """
    a = a.contiguous()
    flat_2d_array = flat_2d_array.contiguous()

    vl = 63
    LEN_1D = a.shape[0]
    n_elements = LEN_1D - 1

    if n_elements <= 0:
        return flat_2d_array

    # Process in strips of 64 elements
    STRIP_SIZE = 64
    BLOCK_SIZE = 256

    # Launch ONE kernel with enough threads for one strip
    # Loop over strips is INSIDE the kernel
    grid_size = triton.cdiv(STRIP_SIZE, BLOCK_SIZE)

    s424_kernel_strip[(grid_size,)](
        flat_2d_array,
        a,
        n_elements,
        vl,
        STRIP_SIZE=STRIP_SIZE,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return flat_2d_array
