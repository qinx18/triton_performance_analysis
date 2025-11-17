"""
Generate Triton implementations using Claude API from baseline code
"""

import anthropic
import os
import sys
from pathlib import Path
from datetime import datetime

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=API_KEY)

def generate_triton_from_baseline(baseline_file, kernel_name):
    """Generate Triton implementation from baseline code"""

    with open(baseline_file, 'r') as f:
        baseline_code = f.read()

    prompt = f"""I have a PyTorch/Python baseline implementation that I want to optimize using Triton for GPU acceleration.

Baseline Implementation:
```python
{baseline_code}
```

Please generate a complete, optimized Triton implementation that:
1. Includes a @triton.jit kernel function named `{kernel_name}_kernel`
2. Includes a Python wrapper function named `{kernel_name}_triton` (NOT `{kernel_name}_pytorch`)
3. The wrapper function must have THE EXACT SAME function signature as the baseline (same parameter names and order)
4. Uses appropriate block sizes and memory access patterns
5. Handles edge cases with masking
6. Is functionally equivalent to the baseline (same inputs, same outputs)
7. Returns the same type as the baseline (single tensor or tuple of tensors)
8. Includes brief comments explaining key optimizations

IMPORTANT:
- Wrapper function name: `{kernel_name}_triton`
- Function signature must match the baseline exactly (same parameters in same order)
- Accept all parameters the baseline accepts, even if some are modified in-place or used only as outputs

Provide ONLY the Python code, no additional explanation."""

    print(f"Generating Triton code for {kernel_name}...")

    model_name = "claude-sonnet-4-20250514"
    message = client.messages.create(
        model=model_name,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )

    # Build complete response with metadata
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    complete_response = f"""# LLM-Generated Triton Implementation for {kernel_name}
# Generated: {timestamp}
# Model: {model_name}
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
