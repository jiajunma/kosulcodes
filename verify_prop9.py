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
            h_w = hrb.canonical_basis_element(w_key)
            lhs = hrb.right_action_H_underline_simple(h_w, i)
            
            lhs_c = hrb.H_to_C(lhs)

            rhs = {}
            # The right side is also computed in canonical basis basis 
            if hrb.is_in_Phi_i(w_key, i):
               rhs[w_key] = -(v**(-1) + v**(1)) 

            else:
                w_star_si = hrb.wtilde_star_si(w_key, i)
                rhs[w_star_si] = sp.Integer(1) 
                for w_prime_key, mu_val in hrb.mu.get(w_key, {}).items():
                    if hrb.is_in_Phi_i(w_prime_key, i):
                        rhs[w_prime_key] = mu_val 
                
            rhs_c = {k: sp.expand(c) for k, c in rhs.items() if sp.expand(c) != 0}
            
            total_checked += 1
            if hrb.is_equal(lhs_c, rhs_c):
                total_passed += 1
            else:
                print(f"-"*60)
                print(f"  FAILED for {hrb.format_element({w_key: sp.Integer(1)}, use_C_basis=True)}, i={i}")
                print(f"  Is in Phi_i: {hrb.is_in_Phi_i(w_key, i)}")
                root_type, companions = hrb._basis[w_key][i]
                print(f"  Root Type: {root_type}")
                # Format companions list properly - join elements with ", "
                comp_strs = [hrb.format_element({c: sp.Integer(1)}, use_C_basis=True) for c in companions]
                print(f"  Companions: [{', '.join(comp_strs)}]")
                if not hrb.is_in_Phi_i(w_key,i):
                    print(f"  w~*s_i: {hrb.format_element({w_star_si: sp.Integer(1)}, use_C_basis=True)}")
                print(f"  LHS_C: {hrb.format_element(lhs_c, use_C_basis=True)}")
                print(f"  RHS_C: {hrb.format_element(rhs_c, use_C_basis=True)}")
                
    print(f"Proposition 9 Passed for  {total_passed}/{total_checked}")

if __name__ == "__main__":
    n = 3
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    test_prop9(n)
