import sys
import sympy as sp
from HeckeRB import HeckeRB
from HeckeRB import v
from RB import denormalize_key

def test_prop9(n):
    print(f"Testing Proposition 9 for n={n}...")
    hrb = HeckeRB(n)
    
    # Pre-compute everything
    print("Pre-computing KL polynomials and mu coefficients...")
    hrb.compute_kl_polynomials()
    hrb.compute_mu_coefficients()
    
    total_checked = 0
    total_passed = 0
    
    for w_key in hrb.basis():
        w, sigma = denormalize_key(w_key)
        for i in range(1, n):
            total_checked += 1
            is_valid = hrb.verify_proposition_9(w_key, i)
            if is_valid:
                total_passed += 1
            else:
                print(f"FAILED: Proposition 9 for w={w}, sigma={set(sigma)}, i={i}")
                # Print LHS and RHS
                h_w = hrb.canonical_basis_element(w_key)
                h_w_H = hrb.H_to_T(h_w)
                lhs = hrb.right_action_H_underline_simple(h_w, i)
                lhs = hrb.T_to_H(lhs)
                
                rhs = {}
                if hrb.is_in_Phi_i(w_key, i):
                    factor = -(v**-1 + v)
                    for k, c in h_w.items():
                        rhs[k] = sp.expand(factor * c)
                else:
                    w_star_si = hrb.wtilde_star_si(w_key, i)
                    h_w_star = hrb.canonical_basis_element(w_star_si)
                    for k, c in h_w_star.items():
                        rhs[k] = rhs.get(k, sp.Integer(0)) + c
                    
                    for w_prime_key, mu_val in hrb.mu.get(w_key, {}).items():
                        if hrb.is_in_Phi_i(w_prime_key, i):
                            h_w_prime = hrb.canonical_basis_element(w_prime_key)
                            for k, c in h_w_prime.items():
                                rhs[k] = rhs.get(k, sp.Integer(0)) + mu_val * c
                
                rhs = {k: sp.expand(c) for k, c in rhs.items() if sp.expand(c) != 0}
                
                print(f"  Is in Phi_i: {hrb.is_in_Phi_i(w_key, i)}")
                print(f"  LHS: {hrb.format_element(lhs)}")
                print(f"  RHS: {hrb.format_element(rhs)}")
                
                # Only show one failure for now
    print(f"Proposition 9 Passed for  {total_passed}/{total_checked}")

if __name__ == "__main__":
    n = 3
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    test_prop9(n)
