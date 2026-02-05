import sys
import sympy as sp
import subprocess
from HeckeRB import HeckeRB, denormalize_key, normalize_key, v
from perm import simple_reflection, permutation_prod, length_of_permutation
from RB import tilde_inverse_sigma

class WGraphRB:
    def __init__(self, n):
        self.n = n
        self.hrb = HeckeRB(n)
        self.hrb.compute_kl_polynomials()
        self.hrb.compute_mu_coefficients()
        self._vertices = list(self.hrb.basis())
        self._tau_map = {}
        for v_key in self._vertices:
            w, sigma = denormalize_key(v_key)
            w_inv, sigma_inv = tilde_inverse_sigma(w, sigma)
            self._tau_map[v_key] = normalize_key(w_inv, sigma_inv)
            
        self._edges = None
        self._double_cells = None
        self._tau_R = None
        self._tau_L = None

    def compute_descents(self):
        if self._tau_R is not None:
            return self._tau_R, self._tau_L
        
        self._tau_R = {}
        self._tau_L = {}
        
        for v_key in self._vertices:
            # Right descents
            dR = set()
            for i in range(1, self.n):
                if self.hrb.is_in_Phi_i(v_key, i):
                    dR.add(i)
            self._tau_R[v_key] = dR
            
            # Left descents: i in tau_L(v) iff i in tau_R(tau(v))
            v_tau_key = self._tau_map[v_key]
            dL = set()
            for i in range(1, self.n):
                if self.hrb.is_in_Phi_i(v_tau_key, i):
                    dL.add(i)
            self._tau_L[v_key] = dL
            
        return self._tau_R, self._tau_L

    
    def compute_edges_right(self,w_key,label_key):
        """
        Compute the edge for right action
        then draw the terms from label_key to the target nodes. 
        """
        edges = []  
        for i in range(1, self.n):
            # use i as the label for right action
            # use -i as the label for left action
            if w_key == label_key:
                label = i
            else:
                label = -i 

            if self.hrb.is_in_Phi_i(w_key, i):
                continue
            else:
                w_star_si = self.hrb.wtilde_star_si(w_key, i)
                w_root_type, w_companions = self.hrb._basis[w_key][i]
                w_star_si_root_type, _ = self.hrb._basis[w_star_si][i]
                # Check if both w and w_star_si are of type T-
                if w_root_type == 'T-' and w_star_si_root_type == 'T-':
                    # Find the type T+ member in the companions of (w, s_i)
                    for companion_key in w_companions:
                        comp_root_type, _ = self.hrb._basis[companion_key][i]
                        if comp_root_type == 'T+':
                            edges.append((label_key, companion_key, label))
                else:
                    edges.append((label_key, w_star_si, label))
                for w_prime_key, mu_val in self.hrb.mu.get(w_key, {}).items():
                    if self.hrb.is_in_Phi_i(w_prime_key, i):
                        edges.append((label_key, w_prime_key, label)) 
        return edges

    def compute_edges(self):
        if self._edges is not None:
            return self._edges
        
        self.compute_descents()
        self._edges = {}  # Dictionary: (source, target) -> list of labels
        print(f"Computing preorder edges from action for n={self.n}...")

        for w_key in self._vertices:
            redges = self.compute_edges_right(w_key, w_key)
            for source, target, label in redges:
                edge = (source, target)
                if edge not in self._edges:
                    self._edges[edge] = []
                if label not in self._edges[edge]:
                    self._edges[edge].append(label)
            
            w_inv_key = normalize_key(*tilde_inverse_sigma(*w_key))  
            ledges = self.compute_edges_right(w_inv_key, w_key)
            for source, target, label in ledges:
                edge = (source, target)
                if edge not in self._edges:
                    self._edges[edge] = []
                if label not in self._edges[edge]:
                    self._edges[edge].append(label)
        
        return self._edges



    def compute_cell(self):
        """
        Compute double cells (strongly connected components) from the preorder edges.
        A double cell is the smallest mutually reachable subset of nodes.
        """
        if self._double_cells is not None:
            return
        
        # First compute all edges
        self.compute_edges()
        
        # Build adjacency list from edges
        from collections import defaultdict
        
        adj = defaultdict(set)
        
        for (source, target), _ in self._edges.items():
            adj[source].add(target)
        
        # Tarjan's algorithm for strongly connected components
        index_counter = [0]
        stack = []
        lowlinks = {}
        index = {}
        on_stack = defaultdict(bool)
        sccs = []
        
        def strongconnect(v):
            index[v] = index_counter[0]
            lowlinks[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack[v] = True
            
            for w in adj[v]:
                if w not in index:
                    strongconnect(w)
                    lowlinks[v] = min(lowlinks[v], lowlinks[w])
                elif on_stack[w]:
                    lowlinks[v] = min(lowlinks[v], index[w])
            
            if lowlinks[v] == index[v]:
                component = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    component.append(w)
                    if w == v:
                        break
                sccs.append(frozenset(component))
        
        for v in self._vertices:
            if v not in index:
                strongconnect(v)
        
        self._double_cells = sccs

    def print_summary(self):
        """Print a summary of the W-graph structure."""
        print(f"\nW-graph summary for HeckeRB(n={self.n})")
        print("="*70)
        print(f"Total vertices: {len(self._vertices)}")
        
        self.compute_cell()
        tau_R, tau_L = self.compute_descents()
        
        print(f"Double cells: {len(self._double_cells)}")
        
        for idx, cell in enumerate(self._double_cells, 1):
            print(f"\nDouble Cell {idx} (size {len(cell)}):")
            print("-"*50)
            
            # Print elements with descents
            for v_key in sorted(cell, key=lambda k: (self.hrb.ell_wtilde(k), k)):
                v_str = self.hrb.format_element({v_key: 1}, use_H_basis=False)
                dR = sorted(list(tau_R[v_key]))
                dL = sorted(list(tau_L[v_key]))
                print(f"    {v_str:20} | dL={dL}, dR={dR}")

    def generate_dot(self):
        """Generate DOT format string for the W-graph (right action)."""
        import re
        
        self.compute_cell()
        self.compute_edges()
        tau_R, tau_L = self.compute_descents()
        
        # Mapping from vertex to cell IDs for coloring
        v_to_double = {}
        for idx, cell in enumerate(self._double_cells):
            for v_el in cell:
                v_to_double[v_el] = idx
        
        # Precompute vertex IDs for performance
        v_to_id = {v_key: f"v{i}" for i, v_key in enumerate(self._vertices)}
        
        lines = [f'digraph WGraphRB_n{self.n} {{']
        lines.append(f'    label="W-graph for n={self.n}, Double Cells: {len(self._double_cells)}";')
        lines.append('    labelloc="t";')
        lines.append('    fontsize=20;')
        lines.append('    rankdir=TB;')
        lines.append('    splines=line;')
        lines.append('    node [shape=box, fontname="Courier", style=filled];')
        lines.append('    edge [fontname="Courier"];')
        
        # Enhanced color palette for double cells with more distinct colors
        colors = [
            "#FFB6C1", "#87CEEB", "#98FB98", "#FFD700", "#FFA07A", 
            "#DDA0DD", "#F0E68C", "#B0E0E6", "#FFE4B5", "#D8BFD8",
            "#FFDAB9", "#E0BBE4", "#C7CEEA", "#FFDFD3", "#B2E2F2",
            "#FFCCF9", "#C9E4CA", "#FFF4E6", "#D5AAFF", "#AED9E0"
        ]
        
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
            lines.append(f'        label="ell={ell}"; style=dashed;')
            for v_key in length_groups[ell]:
                v_id = v_to_id[v_key]
                
                # Format the label with colored permutation using HTML-like labels in DOT
                w, sigma = denormalize_key(v_key)
                ell = self.hrb.ell_wtilde(v_key)
                
                # Format w with colored digits for DOT HTML labels
                # Elements at positions in sigma are red, others are blue
                colored_w_parts = []
                for i in range(1, self.n + 1):
                    val = w[i-1]
                    color = "red" if i in sigma else "blue"
                    colored_w_parts.append(f'<font color="{color}">{val}</font>')
                colored_w_str = "".join(colored_w_parts)
                
                label = f"[{colored_w_str}]"
                
                dR = sorted(list(tau_R[v_key]))
                dL = sorted(list(tau_L[v_key]))
                label += f"<br/><font point-size='10'>L:{dL} R:{dR}</font>"
                
                # Assign color based on double cell membership
                double_idx = v_to_double.get(v_key, 0)
                color = colors[double_idx % len(colors)]
                # Use label=<> for HTML-like labels in DOT
                lines.append(f'        {v_id} [label=<{label}>, fillcolor="{color}"];')
            lines.append('    }')
            
        for (y_key, w_key), labels in self._edges.items():
            y_id = v_to_id[y_key]
            w_id = v_to_id[w_key]
            label_str = ",".join(str(lbl) for lbl in sorted(labels))
            lines.append(f'    {y_id} -> {w_id} [label="{label_str}"];')
            
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
    filename = "WGraphRB.svg"
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            pass
    if len(sys.argv) > 2:
        filename = sys.argv[2]
        
    wg = WGraphRB(n)
    wg.print_summary()
    wg.save_svg(filename)
