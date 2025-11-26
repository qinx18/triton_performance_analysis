"""
Generate Triton implementations using Claude API from baseline code.

Includes both flow dependency (RAW) and WAR anti-dependency analysis
to correctly handle race conditions in parallel execution.
"""

import anthropic
import os
import sys
import re
from pathlib import Path
from datetime import datetime

# Add PET analysis directory to path for WAR analysis
sys.path.insert(0, "/home/qinxiao/workspace/pet/isl_analysis")
from compute_war_dependences import analyze_kernel_war

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=API_KEY)

# Paths to PET analysis
DEPENDENCY_ANALYSIS_FILE = "/home/qinxiao/workspace/pet/isl_analysis/results/flow_deps_s000_s119.txt"
KERNELS_DIR = "/home/qinxiao/workspace/pet/isl_analysis/kernels"

def load_dependency_analysis(kernel_name):
    """Load dependency analysis for a specific kernel from PET analysis file."""
    if not os.path.exists(DEPENDENCY_ANALYSIS_FILE):
        return None

    with open(DEPENDENCY_ANALYSIS_FILE, 'r') as f:
        content = f.read()

    # Split by kernel sections
    # Pattern: ========================================\nKernel: <name>\n========================================
    pattern = rf'={40}\nKernel: {re.escape(kernel_name)}\n={40}\n(.*?)(?=\n={40}\nKernel:|\n={80}|$)'
    match = re.search(pattern, content, re.DOTALL)

    if match:
        return match.group(1).strip()
    return None


def load_war_analysis(kernel_name):
    """Load WAR anti-dependency analysis for a kernel."""
    kernel_file = os.path.join(KERNELS_DIR, f"{kernel_name}.c")
    if not os.path.exists(kernel_file):
        return None

    try:
        return analyze_kernel_war(kernel_file)
    except Exception as e:
        print(f"  Warning: WAR analysis failed: {e}")
        return None


def build_war_instructions(kernel_name, war_result):
    """Build specific instructions for handling WAR dependencies."""
    if not war_result or war_result['parallelization_safe']:
        return ""

    arrays_to_copy = war_result['arrays_needing_copy']

    instructions = f"""
## CRITICAL: WAR Race Condition Handling Required

This kernel has WAR (Write-After-Read) anti-dependencies that cause race conditions in parallel execution.
**Arrays requiring read-only copy**: {arrays_to_copy}

### Problem
In parallel execution across GPU blocks, if block B writes to a location before block A reads from it,
block A reads the wrong (modified) value instead of the original value.

For example, in `a[i+1] = a[i] + b[i]`:
- Thread processing i=100 reads a[100], writes a[101]
- Thread processing i=101 reads a[101], writes a[102]
- If the write to a[101] happens before the read from a[101], incorrect results occur.

### Required Solution: Read-Only Copy Pattern
Pass a **read-only copy** of the array to the kernel. All threads load from the copy (immutable)
and store results to the original array.

### Implementation Template
```python
# In wrapper function - create read-only copy BEFORE launching kernel:
def {kernel_name}_triton(...):
"""
    for arr in arrays_to_copy:
        instructions += f"    {arr}_copy = {arr}.clone()  # Read-only copy\n"

    instructions += f"""
    # Pass BOTH original (for writes) AND copy (for reads) to kernel
    {kernel_name}_kernel[grid](
"""
    for arr in arrays_to_copy:
        instructions += f"        {arr},        # Write destination (original)\n"
        instructions += f"        {arr}_copy,   # Read source (immutable copy)\n"

    instructions += """        ...other_args...
    )

# In kernel - LOAD from copy, STORE to original:
@triton.jit
def {kernel_name}_kernel(
"""
    for arr in arrays_to_copy:
        instructions += f"    {arr}_ptr,        # Write destination\n"
        instructions += f"    {arr}_copy_ptr,   # Read source\n"

    instructions += """    ...
):
    offsets = ...
"""
    for arr in arrays_to_copy:
        instructions += f"    {arr}_vals = tl.load({arr}_copy_ptr + offsets, mask=mask)  # Read from COPY\n"

    instructions += """
    result = ...  # Compute using loaded values

"""
    for arr in arrays_to_copy:
        instructions += f"    tl.store({arr}_ptr + write_offsets, result, mask=mask)  # Write to ORIGINAL\n"

    instructions += """```

**Why this works**: The copy preserves original values for all threads. Reads from immutable copy,
writes to original - no conflict possible.
"""
    return instructions


def generate_triton_from_baseline(baseline_file, kernel_name):
    """Generate Triton implementation from baseline code"""

    with open(baseline_file, 'r') as f:
        baseline_code = f.read()

    # Load flow dependency analysis from PET
    dep_analysis = load_dependency_analysis(kernel_name)
    dep_section = ""
    if dep_analysis:
        dep_section = f"""
## Flow Dependency Analysis (RAW - from Polyhedral Extraction Tool):
```
{dep_analysis}
```
"""

    # Load WAR anti-dependency analysis
    war_result = load_war_analysis(kernel_name)
    war_section = build_war_instructions(kernel_name, war_result)

    prompt = f"""I have a PyTorch/Python baseline implementation that I want to optimize using Triton for GPU acceleration.

## Baseline Implementation:
```python
{baseline_code}
```
{dep_section}{war_section}

## Requirements:
Please generate a complete, optimized Triton implementation that:
1. Includes a @triton.jit kernel function named `{kernel_name}_kernel`
2. Includes a Python wrapper function named `{kernel_name}_triton` (NOT `{kernel_name}_pytorch`)
3. The wrapper function must have THE EXACT SAME function signature as the baseline (same parameter names and order)
4. Uses appropriate block sizes and memory access patterns
5. Handles edge cases with masking
6. Is functionally equivalent to the baseline (same inputs, same outputs)
7. Returns the same type as the baseline (single tensor or tuple of tensors)
8. **CRITICAL**: If WAR dependencies are shown above, you MUST use the read-only copy pattern

IMPORTANT:
- Wrapper function name: `{kernel_name}_triton`
- Function signature must match the baseline exactly (same parameters in same order)
- Accept all parameters the baseline accepts, even if some are modified in-place or used only as outputs
- If WAR analysis shows arrays needing copies, you MUST clone those arrays and pass both original and copy to the kernel

Provide ONLY the Python code, no additional explanation."""

    print(f"Generating Triton code for {kernel_name}...")
    if war_result and not war_result['parallelization_safe']:
        print(f"  ⚠ WAR dependencies detected for arrays: {war_result['arrays_needing_copy']}")

    model_name = "claude-sonnet-4-20250514"
    message = client.messages.create(
        model=model_name,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )

    # Build complete response with metadata
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    war_info = "None" if not war_result or war_result['parallelization_safe'] else str(war_result['arrays_needing_copy'])
    complete_response = f"""# LLM-Generated Triton Implementation for {kernel_name}
# Generated: {timestamp}
# Model: {model_name}
# WAR Dependencies: {war_info}
# Stop Reason: {message.stop_reason}
# Input Tokens: {message.usage.input_tokens}
# Output Tokens: {message.usage.output_tokens}

{'='*80}
PROMPT:
{'='*80}
{prompt}

{'='*80}
RESPONSE:
{'='*80}
{message.content[0].text}
"""

    # Extract just the code content
    response_text = message.content[0].text

    # Clean up code blocks for the code file
    triton_code = response_text
    if "```python" in triton_code:
        triton_code = triton_code.split("```python")[1].split("```")[0].strip()
    elif "```" in triton_code:
        triton_code = triton_code.split("```")[1].split("```")[0].strip()

    return triton_code, complete_response

def main():
    baselines_dir = Path("baselines")
    llm_triton_dir = Path("llm_triton")
    llm_triton_dir.mkdir(exist_ok=True)

    # Create subdirectory for original responses
    raw_responses_dir = llm_triton_dir / "raw_responses"
    raw_responses_dir.mkdir(exist_ok=True)

    # Check for command-line argument
    if len(sys.argv) > 1:
        # Process specific kernel only
        kernel_name = sys.argv[1]
        baseline_file = baselines_dir / f"{kernel_name}_baseline.py"

        if not baseline_file.exists():
            print(f"❌ Error: Baseline file not found: {baseline_file}")
            print(f"Available baselines:")
            for f in sorted(baselines_dir.glob("*_baseline.py")):
                print(f"  - {f.stem.replace('_baseline', '')}")
            return

        print(f"Processing single kernel: {kernel_name}")
        files_to_process = [baseline_file]
    else:
        # Process all baseline files
        print("Processing all baseline files...")
        files_to_process = list(baselines_dir.glob("*.py"))

    for baseline_file in files_to_process:
        kernel_name = baseline_file.stem.replace("_baseline", "")

        try:
            triton_code, complete_response = generate_triton_from_baseline(baseline_file, kernel_name)

            # Save cleaned code
            output_file = llm_triton_dir / f"{kernel_name}_triton_llm.py"
            with open(output_file, 'w') as f:
                f.write(triton_code)

            # Save complete response with metadata
            raw_file = raw_responses_dir / f"{kernel_name}_raw_response.txt"
            with open(raw_file, 'w') as f:
                f.write(complete_response)

            print(f"  ✓ Saved code to: {output_file}")
            print(f"  ✓ Saved complete response with metadata to: {raw_file}\n")

        except Exception as e:
            print(f"  ✗ Error: {e}\n")

    print("✅ LLM Triton generation complete!")

if __name__ == "__main__":
    main()
