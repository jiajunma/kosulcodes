"""
W-graph visualization for the mirabolic Hecke bimodule R.

The W-graph encodes the structure of the Kazhdan-Lusztig basis:
- Vertices: basis elements w̃ = (w, β)
- Edges: pairs (ỹ, w̃) where μ(ỹ, w̃) ≠ 0
- Descent sets: τ(w̃) = {i : w̃ * s_i has lower length}

Usage:
    python WGraph.py <n> [options]
    
Options:
    --dot       : output DOT format for Graphviz
    --svg       : generate SVG file (requires graphviz)
    --text      : text-based output (default)
    --descents  : show descent sets for each vertex
    --matrix    : show adjacency matrix
"""

import sys
import subprocess
from HeckeRB import HeckeRB, denormalize_key, normalize_key
from perm import simple_reflection, permutation_prod, length_of_permutation
from RB import right_action_rb


class WGraph:
    """W-graph for the mirabolic Hecke bimodule R."""
    
    def __init__(self, n):
        self.n = n
        self.R = HeckeRB(n)
        self._vertices = list(self.R.basis())
        self._edges = None
        self._descents = None
        
    def vertices(self):
        """Return all vertices (basis elements)."""
        return self._vertices
    
    def compute_descents(self):
        """
        Compute descent sets τ(w̃) for all vertices.
        
        τ(w̃) = {i : ℓ(w̃ * s_i) < ℓ(w̃)}
        """
        if self._descents is not None:
            return self._descents
        
        self._descents = {}
        for key in self._vertices:
            w, beta = denormalize_key(key)
            ell_w = self.R.ell_wtilde(w, beta)
            
            descents = set()
            for i in range(1, self.n):
                # Compute w̃ * s_i
                s = simple_reflection(i, self.n)
                ws, ws_beta = right_action_rb((w, beta), s)
                ell_ws = self.R.ell_wtilde(ws, ws_beta)
                
                if ell_ws < ell_w:
                    descents.add(i)
            
            self._descents[key] = descents
        
        return self._descents
    
    def compute_edges(self):
        """
        Compute all W-graph edges.
        
        An edge exists from ỹ to w̃ if μ(ỹ, w̃) ≠ 0.
        """
        if self._edges is not None:
            return self._edges
        
        # First compute canonical basis to get mu coefficients
        self.R.compute_canonical_basis_iterative()
        
        self._edges = []
        for w_key in self._vertices:
            lower = self.R.bruhat_lower_elements(w_key)
            for y_key in lower:
                mu = self.R.mu_coefficient(y_key, w_key)
                if mu != 0:
                    self._edges.append((y_key, w_key, mu))
        
        return self._edges
    
    def format_vertex(self, key):
        """Format a vertex for display."""
        w, beta = denormalize_key(key)
        w_str = "".join(str(x) for x in w)
        beta_str = "{" + ",".join(str(x) for x in sorted(beta)) + "}"
        return f"{w_str},{beta_str}"
    
    def format_vertex_short(self, key):
        """Short format for vertex (for DOT labels)."""
        w, beta = denormalize_key(key)
        w_str = "".join(str(x) for x in w)
        if not beta:
            return w_str
        beta_str = "".join(str(x) for x in sorted(beta))
        return f"{w_str}|{beta_str}"
    
    def vertex_id(self, key):
        """Generate a valid DOT node ID."""
        w, beta = denormalize_key(key)
        w_str = "".join(str(x) for x in w)
        beta_str = "".join(str(x) for x in sorted(beta))
        return f"v_{w_str}_{beta_str}"
    
    def print_text(self, show_descents=True):
        """Print W-graph in text format."""
        edges = self.compute_edges()
        descents = self.compute_descents()
        
        print(f"\nW-graph for HeckeRB bimodule (n={self.n})")
        print("=" * 60)
        
        # Vertices with descent sets
        print(f"\nVertices ({len(self._vertices)} total):")
        print("-" * 40)
        
        # Sort by length
        sorted_vertices = sorted(self._vertices,
            key=lambda k: (self.R.ell_wtilde(*denormalize_key(k)), k))
        
        for key in sorted_vertices:
            w, beta = denormalize_key(key)
            ell = self.R.ell_wtilde(w, beta)
            v_str = self.format_vertex(key)
            
            if show_descents:
                tau = descents[key]
                tau_str = "{" + ",".join(str(i) for i in sorted(tau)) + "}"
                print(f"  [{v_str}]  ℓ={ell}  τ={tau_str}")
            else:
                print(f"  [{v_str}]  ℓ={ell}")
        
        # Edges
        print(f"\nEdges ({len(edges)} total):")
        print("-" * 40)
        
        # Sort edges by length difference, then by vertices
        sorted_edges = sorted(edges, key=lambda e: (
            self.R.ell_wtilde(*denormalize_key(e[1])) - 
            self.R.ell_wtilde(*denormalize_key(e[0])),
            e[0], e[1]
        ))
        
        for y_key, w_key, mu in sorted_edges:
            y_str = self.format_vertex(y_key)
            w_str = self.format_vertex(w_key)
            ell_y = self.R.ell_wtilde(*denormalize_key(y_key))
            ell_w = self.R.ell_wtilde(*denormalize_key(w_key))
            print(f"  [{y_str}] --({mu})--> [{w_str}]  (Δℓ={ell_w - ell_y})")
        
        # Statistics
        print(f"\nStatistics:")
        print("-" * 40)
        print(f"  Vertices: {len(self._vertices)}")
        print(f"  Edges: {len(edges)}")
        
        # Count by length
        length_counts = {}
        for key in self._vertices:
            ell = self.R.ell_wtilde(*denormalize_key(key))
            length_counts[ell] = length_counts.get(ell, 0) + 1
        
        print(f"  Vertices by length:")
        for ell in sorted(length_counts.keys()):
            print(f"    ℓ={ell}: {length_counts[ell]}")
    
    def print_matrix(self):
        """Print adjacency matrix of the W-graph."""
        edges = self.compute_edges()
        
        print(f"\nAdjacency matrix for W-graph (n={self.n})")
        print("=" * 60)
        
        # Create edge dict for quick lookup
        edge_dict = {}
        for y_key, w_key, mu in edges:
            edge_dict[(y_key, w_key)] = mu
        
        # Sort vertices by length
        sorted_vertices = sorted(self._vertices,
            key=lambda k: (self.R.ell_wtilde(*denormalize_key(k)), k))
        
        # Print header
        print("\nVertex indices:")
        for i, key in enumerate(sorted_vertices):
            v_str = self.format_vertex(key)
            print(f"  {i}: [{v_str}]")
        
        print("\nMatrix (row y, col w, entry μ(y,w)):")
        n_verts = len(sorted_vertices)
        
        # Print column headers
        header = "    "
        for j in range(n_verts):
            header += f"{j:3d}"
        print(header)
        print("    " + "-" * (3 * n_verts))
        
        # Print rows
        for i, y_key in enumerate(sorted_vertices):
            row = f"{i:3d}|"
            for j, w_key in enumerate(sorted_vertices):
                mu = edge_dict.get((y_key, w_key), 0)
                if mu == 0:
                    row += "  ."
                else:
                    row += f"{mu:3d}"
            print(row)
    
    def generate_dot(self, show_descents=True, rankdir="TB"):
        """
        Generate DOT format for Graphviz visualization.
        
        Args:
            show_descents: include descent sets in labels
            rankdir: graph direction (TB=top-bottom, LR=left-right)
        
        Returns:
            str: DOT format string
        """
        edges = self.compute_edges()
        descents = self.compute_descents()
        
        lines = []
        lines.append(f'digraph WGraph_n{self.n} {{')
        lines.append(f'    rankdir={rankdir};')
        lines.append('    node [shape=box, fontname="Courier"];')
        lines.append('    edge [fontname="Courier"];')
        lines.append('')
        
        # Group vertices by length for ranking
        length_groups = {}
        for key in self._vertices:
            ell = self.R.ell_wtilde(*denormalize_key(key))
            if ell not in length_groups:
                length_groups[ell] = []
            length_groups[ell].append(key)
        
        # Add subgraphs for each length level
        for ell in sorted(length_groups.keys()):
            lines.append(f'    subgraph cluster_ell{ell} {{')
            lines.append(f'        rank=same;')
            lines.append(f'        label="ℓ={ell}";')
            lines.append(f'        style=dashed;')
            
            for key in length_groups[ell]:
                node_id = self.vertex_id(key)
                label = self.format_vertex_short(key)
                
                if show_descents:
                    tau = descents[key]
                    tau_str = ",".join(str(i) for i in sorted(tau))
                    if tau_str:
                        label += f"\\nτ={{{tau_str}}}"
                    else:
                        label += "\\nτ={}"
                
                lines.append(f'        {node_id} [label="{label}"];')
            
            lines.append('    }')
            lines.append('')
        
        # Add edges
        lines.append('    // Edges')
        for y_key, w_key, mu in edges:
            y_id = self.vertex_id(y_key)
            w_id = self.vertex_id(w_key)
            
            if mu == 1:
                lines.append(f'    {y_id} -> {w_id};')
            else:
                lines.append(f'    {y_id} -> {w_id} [label="{mu}"];')
        
        lines.append('}')
        return '\n'.join(lines)
    
    def save_dot(self, filename):
        """Save DOT format to file."""
        dot = self.generate_dot()
        with open(filename, 'w') as f:
            f.write(dot)
        print(f"DOT file saved to: {filename}")
    
    def save_svg(self, filename):
        """
        Generate SVG using Graphviz.
        
        Requires graphviz to be installed (dot command).
        """
        dot = self.generate_dot()
        
        try:
            result = subprocess.run(
                ['dot', '-Tsvg'],
                input=dot,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"Error running graphviz: {result.stderr}")
                return False
            
            with open(filename, 'w') as f:
                f.write(result.stdout)
            
            print(f"SVG file saved to: {filename}")
            return True
            
        except FileNotFoundError:
            print("Error: graphviz (dot) not found. Install with:")
            print("  brew install graphviz  (macOS)")
            print("  apt install graphviz   (Ubuntu)")
            return False
    
    def print_descent_sets(self):
        """Print descent sets for all vertices."""
        descents = self.compute_descents()
        
        print(f"\nDescent sets τ(w̃) for W-graph (n={self.n})")
        print("=" * 60)
        
        # Sort by length
        sorted_vertices = sorted(self._vertices,
            key=lambda k: (self.R.ell_wtilde(*denormalize_key(k)), k))
        
        # Group by descent set
        by_descent = {}
        for key in sorted_vertices:
            tau = frozenset(descents[key])
            if tau not in by_descent:
                by_descent[tau] = []
            by_descent[tau].append(key)
        
        print("\nVertices grouped by descent set:")
        print("-" * 40)
        
        for tau in sorted(by_descent.keys(), key=lambda t: (len(t), tuple(sorted(t)))):
            tau_str = "{" + ",".join(str(i) for i in sorted(tau)) + "}"
            vertices = by_descent[tau]
            v_strs = [self.format_vertex(k) for k in vertices]
            print(f"  τ = {tau_str}:")
            for v_str in v_strs:
                print(f"    [{v_str}]")
        
        print("\nDescent statistics:")
        print("-" * 40)
        for i in range(1, self.n):
            count = sum(1 for k in self._vertices if i in descents[k])
            print(f"  s_{i} is a descent for {count}/{len(self._vertices)} vertices")


def main():
    if len(sys.argv) < 2:
        print("Usage: python WGraph.py <n> [options]")
        print("  n: size of the symmetric group")
        print("  options:")
        print("    --text      : text-based output (default)")
        print("    --descents  : show descent sets for each vertex")
        print("    --matrix    : show adjacency matrix")
        print("    --dot       : output DOT format for Graphviz")
        print("    --svg       : generate SVG file (requires graphviz)")
        print("    --all       : show all outputs")
        sys.exit(1)
    
    n = int(sys.argv[1])
    
    show_text = "--text" in sys.argv or len(sys.argv) == 2
    show_descents = "--descents" in sys.argv
    show_matrix = "--matrix" in sys.argv
    show_dot = "--dot" in sys.argv
    show_svg = "--svg" in sys.argv
    show_all = "--all" in sys.argv
    
    if show_all:
        show_text = show_descents = show_matrix = show_dot = show_svg = True
    
    print(f"\n=== W-graph for HeckeRB bimodule (n={n}) ===")
    
    G = WGraph(n)
    
    if show_text or (not show_descents and not show_matrix and not show_dot and not show_svg):
        G.print_text(show_descents=True)
    
    if show_descents:
        G.print_descent_sets()
    
    if show_matrix:
        G.print_matrix()
    
    if show_dot:
        dot = G.generate_dot()
        print("\nDOT format:")
        print("-" * 40)
        print(dot)
        
        # Also save to file
        G.save_dot(f"WGraph_n{n}.dot")
    
    if show_svg:
        G.save_svg(f"WGraph_n{n}.svg")


if __name__ == "__main__":
    main()
