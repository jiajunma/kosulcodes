import sys
import subprocess
import sympy as sp

from HeckeBessel import HeckeBessel
from verify_bar_bessel import compute_bar_involution_bessel_id
from RB import denormalize_key, tilde_inverse_sigma


class WGraphBessel2:
    """
    Bessel W-graph variant (canonical basis multiplication).
    Draw an edge w -> y if C_y appears in C_w * H_i or H_i * C_w.
    """

    def __init__(self, n):
        self.n = n
        self.hrb = HeckeBessel(n, strict=True)
        # Ensure bar table uses Bessel initialization
        self.hrb.bar_table = compute_bar_involution_bessel_id(self.hrb, verbose=False)
        # Ensure KL polynomials and inverse KL are computed for H_to_C
        self.hrb.compute_kl_polynomials()
        self.hrb.compute_inverse_kl_polynomials()
        self._vertices = list(self.hrb.basis())
        self._edges = {}  # (source_key, target_key) -> set of reflection indices
        self._tau_R = None
        self._tau_L = None
        self._tau_map = {}
        for v_key in self._vertices:
            w, sigma = denormalize_key(v_key)
            w_inv, sigma_inv = tilde_inverse_sigma(w, sigma)
            self._tau_map[v_key] = (tuple(w_inv), frozenset(sigma_inv))

    def compute_descents(self):
        if self._tau_R is not None:
            return self._tau_R, self._tau_L

        self._tau_R = {}
        self._tau_L = {}

        for v_key in self._vertices:
            dR = set()
            for i in range(1, self.n):
                if self.hrb.is_in_Phi_i(v_key, i):
                    dR.add(i)
            self._tau_R[v_key] = dR

            dL = set()
            # Left descents: use explicit left root types (s_1..s_{n-2})
            if hasattr(self.hrb, "_left_basis") and v_key in self.hrb._left_basis:
                for i in range(1, self.n - 1):
                    T, _ = self.hrb._left_basis[v_key][i]
                    if T in ["G", "U+", "T+"]:
                        dL.add(i)
            self._tau_L[v_key] = dL

        return self._tau_R, self._tau_L

    def compute_edges(self):
        """
        Draw an arrow from w to y if C_y occurs in C_w * H_i or H_i * C_w.
        """
        print(f"Computing Bessel W-graph edges for n={self.n}...")
        for w_key in self._vertices:
            c_w = self.hrb.canonical_basis_element(w_key)

            for i in range(1, self.n):
                # Right action: C_w * H_i
                lhs_h_right = self.hrb.right_action_H_underline_simple(c_w, i)
                lhs_c_right = self.hrb.H_to_C(lhs_h_right)

                for y_key, coeff in lhs_c_right.items():
                    if sp.expand(coeff) == 0 or w_key == y_key:
                        continue
                    edge = (w_key, y_key)
                    if edge not in self._edges:
                        self._edges[edge] = set()
                    self._edges[edge].add(i)

                # Left action: H_i * C_w (only for i <= n-2)
                if i >= self.n - 1:
                    continue
                lhs_h_left = self.hrb.left_action_H_underline_simple(c_w, i)
                lhs_c_left = self.hrb.H_to_C(lhs_h_left)

                for y_key, coeff in lhs_c_left.items():
                    if sp.expand(coeff) == 0 or w_key == y_key:
                        continue
                    edge = (w_key, y_key)
                    if edge not in self._edges:
                        self._edges[edge] = set()
                    self._edges[edge].add(i)

    def compute_cell(self):
        """
        Compute double cells (strongly connected components) from the edges.
        """
        if not self._edges:
            self.compute_edges()

        from collections import defaultdict
        adj = defaultdict(set)
        for (source, target) in self._edges:
            adj[source].add(target)

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
        return sccs

    def generate_dot(self):
        self.compute_edges()
        self.compute_cell()
        tau_R, tau_L = self.compute_descents()

        # Ensure mu coefficients exist for edge labels
        if not hasattr(self.hrb, "mu"):
            self.hrb.compute_mu_coefficients()

        v_to_id = {v_key: f"v{idx}" for idx, v_key in enumerate(self._vertices)}
        v_to_cell = {}
        for idx, cell in enumerate(self._double_cells):
            for v_key in cell:
                v_to_cell[v_key] = idx

        lines = ['digraph WGraphBessel2 {']
        lines.append(f'    label="Bessel W-graph 2 for n={self.n}, Double Cells: {len(self._double_cells)}";')
        lines.append('    labelloc="t";')
        lines.append('    fontsize=20;')
        lines.append('    rankdir=TB;')
        lines.append('    splines=true;')
        lines.append('    node [shape=box, fontname="Courier", style=filled];')
        lines.append('    edge [fontname="Courier", fontsize=10];')

        colors = [
            "#FFB6C1", "#87CEEB", "#98FB98", "#FFD700", "#FFA07A",
            "#DDA0DD", "#F0E68C", "#B0E0E6", "#FFE4B5", "#D8BFD8",
            "#FFDAB9", "#E0BBE4", "#C7CEEA", "#FFDFD3", "#B2E2F2",
            "#FFCCF9", "#C9E4CA", "#FFF4E6", "#D5AAFF", "#AED9E0"
        ]

        for v_key in self._vertices:
            v_id = v_to_id[v_key]
            w, sigma = denormalize_key(v_key)
            ell = self.hrb.ell_wtilde(v_key)

            colored_w_parts = []
            for i in range(1, self.n + 1):
                val = w[i - 1]
                color = "red" if i in sigma else "blue"
                colored_w_parts.append(f'<font color="{color}">{val}</font>')
            colored_w_str = "".join(colored_w_parts)

            label = f"[{colored_w_str}]{ell}"
            dR = sorted(list(tau_R[v_key]))
            dL = sorted(list(tau_L[v_key]))
            label += f"<br/><font point-size='10'>L:{dL} R:{dR}</font>"

            cell_idx = v_to_cell.get(v_key, 0)
            color = colors[cell_idx % len(colors)]
            lines.append(f'    {v_id} [label=<{label}>, fillcolor="{color}"];')

        # Build symmetric mu map for labeling
        mu_sym = {}
        for w_key, row in self.hrb.mu.items():
            for y_key, mu in row.items():
                if mu == 0:
                    continue
                mu_sym[(w_key, y_key)] = mu
                mu_sym[(y_key, w_key)] = mu

        processed_edges = set()
        for (w_key, y_key) in self._edges:
            if (w_key, y_key) in processed_edges:
                continue

            w_id = v_to_id[w_key]
            y_id = v_to_id[y_key]
            mu_val = mu_sym.get((w_key, y_key), 0)
            label_str = f"{mu_val}" if mu_val != 0 else ""

            if (y_key, w_key) in self._edges:
                lines.append(f'    {w_id} -> {y_id} [dir=both, label="{label_str}"];')
                processed_edges.add((y_key, w_key))
            else:
                lines.append(f'    {w_id} -> {y_id} [label="{label_str}"];')

            processed_edges.add((w_key, y_key))

        lines.append('}')
        return "\n".join(lines)

    def save_svg(self, filename):
        dot_str = self.generate_dot()
        try:
            process = subprocess.Popen(
                ['dot', '-Tsvg', '-o', filename],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            _, stderr = process.communicate(input=dot_str)
            if process.returncode == 0:
                print(f"SVG saved to {filename}")
            else:
                print(f"Error generating SVG: {stderr}")
        except FileNotFoundError:
            print("Graphviz 'dot' command not found. Please install Graphviz.")

    def save_pdf(self, filename):
        dot_str = self.generate_dot()
        try:
            process = subprocess.Popen(
                ['dot', '-Tpdf', '-o', filename],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            _, stderr = process.communicate(input=dot_str)
            if process.returncode == 0:
                print(f"PDF saved to {filename}")
            else:
                print(f"Error generating PDF: {stderr}")
        except FileNotFoundError:
            print("Graphviz 'dot' command not found. Please install Graphviz.")


if __name__ == "__main__":
    n = 2
    filename = "WGraphBessel2.svg"
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            pass
    if len(sys.argv) > 2:
        filename = sys.argv[2]

    wg = WGraphBessel2(n)
    wg.compute_cell()

    if filename.lower().endswith('.pdf'):
        wg.save_pdf(filename)
    else:
        wg.save_svg(filename)
