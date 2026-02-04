import sys
import sympy as sp
from HeckeRB import HeckeRB
from RB import denormalize_key

def verify_canonical_bases(n):
    print(f"Initializing HeckeRB for n={n}...")
    R = HeckeRB(n)
    
    print("Computing KL polynomials...")
    R.compute_kl_polynomials(verbose=True)
    
    print("Verifying bar-invariance of canonical basis elements...")
    all_ok = True
    count = 0
    total = len(list(R.basis()))
    
    for key in R.basis():
        count += 1
        w, beta = denormalize_key(key)
        
        # Use colored permutation for presentation
        wtilde_str = R._format_wtilde(w, beta)
        
        # Compute canonical basis element C_w
        # C_w = sum P_{y,w} H_y
        c_w = R.canonical_basis_element((w, beta))
        
        # Compute bar(C_w) using bar_H because coefficients are in H-basis
        #bar_c_w = R.bar_H(c_w)
        bar_c_w = R.bar_H(c_w)
        
        # Check equality
        if R.is_equal(c_w, bar_c_w):
            print(f"[{count}/{total}] C[{wtilde_str}] is bar-invariant \u2713")
        else:
            all_ok = False
            print(f"[{count}/{total}] C[{wtilde_str}] is NOT bar-invariant \u2717")
            # For H-basis elements, use format_element with use_H_basis=True
            print(f"  C_w: {R.format_element(c_w, use_H_basis=True)}")
            print(f"  bar(C_w): {R.format_element(bar_c_w, use_H_basis=True)}")
            
            # Print differences
            diff = {}
            all_keys = set(c_w.keys()) | set(bar_c_w.keys())
            for k in all_keys:
                d = sp.expand(c_w.get(k, 0) - bar_c_w.get(k, 0))
                if d != 0:
                    diff[k] = d
            if diff:
                print(f"  Difference: {R._format_T_element(diff)}")

    if all_ok:
        print("\nSUCCESS: All canonical basis elements are bar-invariant.")
    else:
        print("\nFAILURE: Some canonical basis elements are not bar-invariant.")

if __name__ == "__main__":
    n = 2
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            print(f"Invalid n: {sys.argv[1]}. Using default n=2.")
    
    verify_canonical_bases(n)
