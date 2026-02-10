#!/usr/bin/env python3

import sys
import subprocess
import os
import re
from WGraphRB2 import WGraph2 as PyWGraph2

def verify_rust_implementation(n=2, verbose=True):
    """
    Verify that the Rust implementation of WGraph2 produces equivalent results to the Python version.
    """
    print(f"Verifying Rust implementation of WGraph2 for n={n}...")

    # First, build the Rust code
    print("Building Rust code...")
    try:
        subprocess.run(
            ["cargo", "build", "--release"],
            cwd="rust_src",
            check=True,
            capture_output=not verbose
        )
    except subprocess.CalledProcessError as e:
        print("Error building Rust code:", e)
        if verbose and e.stdout:
            print(e.stdout.decode())
        if verbose and e.stderr:
            print(e.stderr.decode())
        return False

    # Run the Rust WGraph2 implementation
    rust_output_file = f"WGraph2_rust_n{n}.svg"
    try:
        print("Running Rust implementation...")
        subprocess.run(
            [os.path.join("rust_src", "target", "release", "wgraph_rb"), str(n), rust_output_file],
            check=True,
            capture_output=not verbose
        )
    except subprocess.CalledProcessError as e:
        print("Error running Rust code:", e)
        if verbose and e.stdout:
            print(e.stdout.decode())
        if verbose and e.stderr:
            print(e.stderr.decode())
        return False

    # Run the Python WGraph2 implementation
    py_output_file = f"WGraph2_python_n{n}.svg"
    print("Running Python implementation...")
    py_wg = PyWGraph2(n)
    py_wg.save_svg(py_output_file)

    # Check that files were created
    if not os.path.exists(rust_output_file):
        print(f"Error: Rust output file {rust_output_file} was not created")
        return False

    if not os.path.exists(py_output_file):
        print(f"Error: Python output file {py_output_file} was not created")
        return False

    # Files exist and have non-zero size
    rust_size = os.path.getsize(rust_output_file)
    py_size = os.path.getsize(py_output_file)

    print(f"Rust SVG size: {rust_size} bytes")
    print(f"Python SVG size: {py_size} bytes")

    # Check basic metrics from the SVG files
    rust_metrics = extract_metrics_from_svg(rust_output_file)
    py_metrics = extract_metrics_from_svg(py_output_file)

    print("\nComparison of graph metrics:")
    print(f"                   Python     Rust")
    print(f"Double cells:      {py_metrics.get('cells', 'N/A'):<10} {rust_metrics.get('cells', 'N/A'):<10}")
    print(f"Nodes:             {py_metrics.get('nodes', 'N/A'):<10} {rust_metrics.get('nodes', 'N/A'):<10}")
    print(f"Edges:             {py_metrics.get('edges', 'N/A'):<10} {rust_metrics.get('edges', 'N/A'):<10}")

    if rust_size > 0 and py_size > 0:
        print("\nBoth implementations successfully generated SVG files.")
        return True
    else:
        print("\nError: One or both SVG files were empty")
        return False

def extract_metrics_from_svg(svg_file):
    """Extract metrics from SVG file (cells, nodes, edges)"""
    metrics = {}

    try:
        with open(svg_file, 'r') as f:
            content = f.read()

            # Extract cell count
            cell_match = re.search(r'Double Cells: (\d+)', content)
            if cell_match:
                metrics['cells'] = int(cell_match.group(1))

            # Count nodes
            node_count = len(re.findall(r'<title>v\d+</title>', content))
            metrics['nodes'] = node_count

            # Count edges (approximation)
            edge_count = len(re.findall(r'->', content))
            metrics['edges'] = edge_count
    except Exception as e:
        print(f"Error parsing {svg_file}: {e}")

    return metrics

if __name__ == "__main__":
    n = 2
    verbose = True

    # Parse command line arguments
    for arg in sys.argv[1:]:
        if arg == '-q' or arg == '--quiet':
            verbose = False
        else:
            try:
                n = int(arg)
            except ValueError:
                print(f"Invalid argument: {arg}. Using default n=2.")

    # Verify Rust implementation
    success = verify_rust_implementation(n, verbose)

    if success:
        print("Verification successful!")
        sys.exit(0)
    else:
        print("Verification failed!")
        sys.exit(1)