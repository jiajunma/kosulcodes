#!/usr/bin/env python3
"""
Compute the dimension of S_n irreducible representations given by partitions.

The dimension is computed using the hook length formula:
    dim(λ) = n! / ∏(hook lengths)

where the hook length h(i,j) at position (i,j) in the Young diagram is:
    h(i,j) = λ_i - j + λ'_j - i + 1
"""

import sys
from math import factorial


def partition_to_conjugate(partition):
    """
    Compute the conjugate (transpose) of a partition.
    
    Args:
        partition: list of integers [p_1, p_2, ..., p_r] in non-increasing order
        
    Returns:
        list: conjugate partition
    """
    if not partition:
        return []
    
    conjugate = []
    max_length = partition[0]
    
    for col in range(max_length):
        height = sum(1 for row_len in partition if row_len > col)
        conjugate.append(height)
    
    return conjugate


def hook_length(partition, i, j):
    """
    Compute the hook length at position (i, j) in the Young diagram.
    
    Args:
        partition: list of integers [p_1, p_2, ..., p_r]
        i: row index (0-based)
        j: column index (0-based)
        
    Returns:
        int: hook length at (i, j)
    """
    conjugate = partition_to_conjugate(partition)
    
    # h(i,j) = (arm length) + (leg length) + 1
    # arm length = λ_i - j - 1 (boxes to the right)
    # leg length = λ'_j - i - 1 (boxes below)
    
    arm = partition[i] - j - 1
    leg = conjugate[j] - i - 1
    
    return arm + leg + 1


def partition_dimension(partition):
    """
    Compute the dimension of the S_n irreducible representation 
    corresponding to a partition using the hook length formula.
    
    Args:
        partition: list of integers [p_1, p_2, ..., p_r] in non-increasing order
        
    Returns:
        int: dimension of the representation
    """
    if not partition:
        return 1
    
    n = sum(partition)
    
    # Compute product of all hook lengths
    hook_product = 1
    for i, row_len in enumerate(partition):
        for j in range(row_len):
            hook_product *= hook_length(partition, i, j)
    
    return factorial(n) // hook_product


def print_young_diagram(partition):
    """
    Print the Young diagram with hook lengths.
    
    Args:
        partition: list of integers
    """
    print("Young diagram:")
    for i, row_len in enumerate(partition):
        print("  " + "□ " * row_len)
    
    print("\nHook lengths:")
    for i, row_len in enumerate(partition):
        hooks = [str(hook_length(partition, i, j)) for j in range(row_len)]
        print("  " + " ".join(f"{h:>2}" for h in hooks))


def main():
    if len(sys.argv) < 2:
        print("Usage: python sn_dim.py <partition>")
        print("Examples:")
        print("  python sn_dim.py 3,2,1      # partition [3,2,1]")
        print("  python sn_dim.py 4,2         # partition [4,2]")
        print("  python sn_dim.py 5           # partition [5]")
        print("  python sn_dim.py 3,2,1 -v    # verbose mode with diagram")
        sys.exit(1)
    
    # Parse partition from command line
    partition_str = sys.argv[1]
    try:
        partition = [int(x) for x in partition_str.split(',')]
    except ValueError:
        print(f"Error: Invalid partition format '{partition_str}'")
        print("Use comma-separated integers, e.g., '3,2,1'")
        sys.exit(1)
    
    # Validate partition (should be non-increasing)
    for i in range(len(partition) - 1):
        if partition[i] < partition[i + 1]:
            print(f"Error: Partition {partition} is not in non-increasing order")
            sys.exit(1)
    
    # Check for verbose mode
    verbose = '-v' in sys.argv or '--verbose' in sys.argv
    
    n = sum(partition)
    dim = partition_dimension(partition)
    
    print(f"Partition: {partition}")
    print(f"n = {n}")
    print(f"Dimension: {dim}")
    
    if verbose:
        print()
        print_young_diagram(partition)
        print(f"\nFormula: dim = {n}! / (product of hook lengths)")
        print(f"       = {factorial(n)} / {factorial(n) // dim}")
        print(f"       = {dim}")


if __name__ == "__main__":
    main()
