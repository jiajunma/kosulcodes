import sympy as sp
from HeckeRB import HeckeRB
from RB import denormalize_key

def test_inverse_kl(n=2):
    print(f"Testing Inverse KL Polynomials for n={n}...")
    hrb = HeckeRB(n)
    
    # Pre-compute KL polynomials
    print("Computing KL polynomials...")
    hrb.compute_kl_polynomials()
    
    # Compute inverse KL polynomials
    print("Computing inverse KL polynomials...")
    hrb.compute_inverse_kl_polynomials(verbose=True)
    
    basis = list(hrb.basis())
    all_passed = True
    total_checked = 0
    
    print(f"Verifying property: sum_y Q_{{z,y}} P_{{y,x}} = delta_{{z,x}} for {len(basis)} basis elements")
    
    for x in basis:
        for z in basis:
            total_checked += 1
            # Compute sum_y Q_{z,y} P_{y,x}
            # Note: P_{y,x} is non-zero only for y <= x
            #       Q_{z,y} is non-zero only for z <= y
            # So the sum is over z <= y <= x
            
            total_sum = sp.Integer(0)
            for y in basis:
                p_yx = hrb.kl_polynomial(y, x)
                q_zy = hrb.inverse_kl_polynomial(z, y)
                if p_yx != 0 and q_zy != 0:
                    total_sum += q_zy * p_yx
            
            total_sum = sp.expand(total_sum)
            expected = sp.Integer(1) if x == z else sp.Integer(0)
            
            if total_sum != expected:
                w_x, s_x = denormalize_key(x)
                w_z, s_z = denormalize_key(z)
                print(f"FAILED for x=(w={w_x}, s={set(s_x)}), z=(w={w_z}, s={set(s_z)}):")
                print(f"  Expected {expected}, got {total_sum}")
                all_passed = False
                
    if all_passed:
        print(f"SUCCESS: Inverse property verified for n={n} ({total_checked} pairs checked)")
    else:
        print(f"FAILURE: Inverse property verification failed for n={n}")

if __name__ == "__main__":
    import sys
    n = 2
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            pass
    test_inverse_kl(n)
