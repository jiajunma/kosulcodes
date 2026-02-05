from HeckeRB import HeckeRB
from RB import root_type_left, str_colored_partition, denormalize_key
from perm import length_of_permutation

def print_root_types(w, sigma):
    """
    Computes and prints the root types for a given permutation and sigma set.
    """
    # Use colored partition for the main (w, sigma) pair
    colored_w_sigma = str_colored_partition(w, sigma)
    print(f"Pair (w, sigma): {colored_w_sigma}")
    
    try:
        results = root_type_left(w, sigma)
        n = len(w)
        
        print(f"  result[0] (Total depth): {results[0]}")
        
        for i in range(1, n):
            T, C = results[i]
            # Format companions with colored partition for compact printing
            comps_list = [str_colored_partition(cw, cs) for cw, cs in C]
            comps_str = " | ".join(comps_list)
            print(f"  s_{i}: Type {T}, Companions: {comps_str}")
            
    except Exception as e:
        print(f"  Error computing root types: {e}")

if __name__ == "__main__":
    import sys
    
    # Default n=3 if not provided
    n = 3
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            pass
            
    print(f"Outputting all pairs (w, sigma) from HeckeRB basis for n={n}")
    print("=" * 60)
    
    hrb = HeckeRB(n)
    for key in hrb.basis():
        w, sigma = denormalize_key(key)
        print_root_types(w, sigma)
        print("-" * 60)
