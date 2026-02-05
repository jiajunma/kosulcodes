import sys
import subprocess
import re
from HeckeRB import HeckeRB, denormalize_key, normalize_key
from RB import tilde_inverse_sigma

class WGraph2:
    def __init__(self, n):
        self.n = n
        self.hrb = HeckeRB(n)
        # Ensure KL polynomials and inverse KL are computed for H_to_C
        self.hrb.compute_kl_polynomials()
        self.hrb.compute_inverse_kl_polynomials()
        self._vertices = list(self.hrb.basis())
        self._edges = {}  # (source_key, target_key) -> set of reflection indices

    def compute_edges(self):
        """
        Draw an arrow from w to y if C_y occurs in C_w * C_s and add the label s.
        """
        print(f"Computing W-graph edges for n={self.n}...")
        for w_key in self._vertices:
            # Get canonical basis element C_w in H-basis
            c_w = self.hrb.canonical_basis_element(w_key)
            w_inv_key = normalize_key(*tilde_inverse_sigma(*w_key))
            c_w_inv = self.hrb.canonical_basis_element(w_inv_key)

            for i in range(1, self.n):
                # Compute C_w * C_si (H_underline_si) in H-basis
                lhs_h = self.hrb.right_action_H_underline_simple(c_w, i)
                # Convert result to C-basis
                lhs_c = self.hrb.H_to_C(lhs_h)
                
                # For each C_y that appears in the expansion
                for y_key in lhs_c:
                    if w_key == y_key:
                        continue
                    edge = (w_key, y_key)
                    if edge not in self._edges:
                        self._edges[edge] = set()
                    self._edges[edge].add(i)
                lhs_h = self.hrb.right_action_H_underline_simple(c_w_inv, i)
                # Convert result to C-basis
                lhs_c_inv = self.hrb.H_to_C(lhs_h)
                # For each C_y that appears in the expansion
                for y_key in lhs_c_inv:
                    if w_key == y_key:
                        continue
                    edge = (w_key, y_key)
                    if edge not in self._edges:
                        self._edges[edge] = set()
                    self._edges[edge].add(i)
                

    def generate_dot(self):
        """Generate DOT format string for the W-graph."""
        self.compute_edges()
        
        v_to_id = {v_key: f"v{idx}" for idx, v_key in enumerate(self._vertices)}
        
        lines = ['digraph WGraph2 {']
        lines.append('    rankdir=TB;')
        #lines.append('    splines=line;')
        lines.append('    node [shape=box, fontname="Courier", style=filled, fillcolor="#F5F5F5"];')
        lines.append('    edge [fontname="Courier", fontsize=10];')
        
        # Group by length for better layout
        length_groups = {}
        for v_key in self._vertices:
            ell = self.hrb.ell_wtilde(v_key)
            if ell not in length_groups:
                length_groups[ell] = []
            length_groups[ell].append(v_key)
            
        for ell in sorted(length_groups.keys()):
            lines.append(f'    subgraph cluster_ell{ell} {{')
            lines.append('        rank=same;')
            lines.append(f'        label="ell={ell}"; style=dashed; color=gray;')
            for v_key in length_groups[ell]:
                v_id = v_to_id[v_key]
                w, sigma = denormalize_key(v_key)
                
                # Colored permutation digits
                colored_w_parts = []
                for i in range(1, self.n + 1):
                    val = w[i-1]
                    color = "red" if i in sigma else "blue"
                    colored_w_parts.append(f'<font color="{color}">{val}</font>')
                colored_w_str = "".join(colored_w_parts)
                
                label = f"[{colored_w_str}]"
                lines.append(f'        {v_id} [label=<{label}>];')
            lines.append('    }')
            
        # Draw edges
        for (w_key, y_key), indices in self._edges.items():
            w_id = v_to_id[w_key]
            y_id = v_to_id[y_key]
            label_str = ",".join(str(i) for i in sorted(list(indices)))
            label_str = ""
            lines.append(f'    {w_id} -> {y_id} [label="{label_str}"];')
            
        lines.append('}')
        return "\n".join(lines)

    def save_svg(self, filename):
        dot_str = self.generate_dot()
        try:
            process = subprocess.Popen(['dot', '-Tsvg', '-o', filename], 
                                     stdin=subprocess.PIPE, 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE, 
                                     text=True)
            stdout, stderr = process.communicate(input=dot_str)
            if process.returncode == 0:
                print(f"SVG saved to {filename}")
            else:
                print(f"Error generating SVG: {stderr}")
        except FileNotFoundError:
            print("Graphviz 'dot' command not found. Please install Graphviz.")

if __name__ == "__main__":
    n = 2
    filename = "WGraph2.svg"
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            pass
    if len(sys.argv) > 2:
        filename = sys.argv[2]
        
    wg = WGraph2(n)
    wg.save_svg(filename)
