import sys
import subprocess

from HeckeBessel import HeckeBessel
from verify_bar_bessel import compute_bar_involution_bessel_id
from RB import denormalize_key, tilde_inverse_sigma


class WGraphBessel3:
    """
    Bessel W-graph variant (directed, mu-symmetric):
    Add a directed edge w -> w' iff:
      1) mu(w, w') != 0, extended symmetrically by
         mu(w,w') := mu(w',w) when only w' > w is defined.
      2) (L(w') \\ L(w)) ∪ (R(w') \\ R(w)) is non-empty
    """

    def __init__(self, n):
        self.n = n
        self.hrb = HeckeBessel(n, strict=True)
        self.hrb.bar_table = compute_bar_involution_bessel_id(self.hrb, verbose=False)
        self.hrb.compute_kl_polynomials()
        self.hrb.compute_mu_coefficients()
        self._vertices = list(self.hrb.basis())
        self._edges = None
        self._double_cells = None

        self._tau_map = {}
        for v_key in self._vertices:
            w, sigma = denormalize_key(v_key)
            w_inv, sigma_inv = tilde_inverse_sigma(w, sigma)
            self._tau_map[v_key] = (tuple(w_inv), frozenset(sigma_inv))

        self._tau_R = None
        self._tau_L = None
        self._tau_combined = None

    def compute_descents(self):
        if self._tau_combined is not None:
            return self._tau_R, self._tau_L, self._tau_combined

        self._tau_R = {}
        self._tau_L = {}
        self._tau_combined = {}

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

            self._tau_combined[v_key] = dR | dL

        return self._tau_R, self._tau_L, self._tau_combined

    def compute_edges(self):
        if self._edges is not None:
            return self._edges

        tau_R, tau_L, _ = self.compute_descents()
        self._edges = {}

        # mu table stores mu(w_key)[y_key] = mu(y_key, w_key)
        # Build symmetric mu for all ordered pairs.
        mu_sym = {}
        for w_key, row in self.hrb.mu.items():
            for y_key, mu in row.items():
                if mu == 0:
                    continue
                mu_sym[(w_key, y_key)] = mu
                mu_sym[(y_key, w_key)] = mu

        for (src, dst), mu in mu_sym.items():
            if not (tau_R[dst] - tau_R[src]) and not (tau_L[dst] - tau_L[src]):
                continue
            edge = (src, dst)
            if edge not in self._edges:
                self._edges[edge] = []
            if mu not in self._edges[edge]:
                self._edges[edge].append(mu)

        return self._edges

    def compute_cell(self):
        if self._double_cells is not None:
            return

        self.compute_edges()

        from collections import defaultdict

        adj = defaultdict(set)
        for (source, target), _ in self._edges.items():
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

    def generate_dot(self):
        self.compute_cell()
        self.compute_edges()
        tau_R, tau_L, _ = self.compute_descents()

        v_to_double = {}
        for idx, cell in enumerate(self._double_cells):
            for v_el in cell:
                v_to_double[v_el] = idx

        v_to_id = {v_key: f"v{i}" for i, v_key in enumerate(self._vertices)}

        lines = [f'digraph WGraphBessel3_n{self.n} {{']
        lines.append(f'    label="Bessel W-graph 3 (n={self.n}), Double Cells: {len(self._double_cells)}";')
        lines.append('    labelloc="t";')
        lines.append('    fontsize=20;')
        lines.append('    rankdir=TB;')
        lines.append('    splines=true;')
        lines.append('    node [shape=box, fontname="Courier", style=filled];')
        lines.append('    edge [fontname="Courier"];')

        colors = [
            "#FFB6C1", "#87CEEB", "#98FB98", "#FFD700", "#FFA07A",
            "#DDA0DD", "#F0E68C", "#B0E0E6", "#FFE4B5", "#D8BFD8",
            "#FFDAB9", "#E0BBE4", "#C7CEEA", "#FFDFD3", "#B2E2F2",
            "#FFCCF9", "#C9E4CA", "#FFF4E6", "#D5AAFF", "#AED9E0"
        ]

        for v_key in self._vertices:
            v_id = v_to_id[v_key]
            w, sigma = denormalize_key(v_key)

            colored_w_parts = []
            for i in range(1, self.n + 1):
                val = w[i - 1]
                color = "red" if i in sigma else "blue"
                colored_w_parts.append(f'<font color="{color}">{val}</font>')
            colored_w_str = "".join(colored_w_parts)

            label = f"[{colored_w_str}]"
            dR = sorted(list(tau_R[v_key]))
            dL = sorted(list(tau_L[v_key]))
            label += f"<br/><font point-size='10'>L:{dL} R:{dR}</font>"

            double_idx = v_to_double.get(v_key, 0)
            color = colors[double_idx % len(colors)]
            lines.append(f'    {v_id} [label=<{label}>, fillcolor="{color}"];')

        for (src, dst), labels in self._edges.items():
            src_id = v_to_id[src]
            dst_id = v_to_id[dst]
            label_str = ",".join(str(lbl) for lbl in sorted(labels))
            lines.append(f'    {src_id} -> {dst_id} [label="{label_str}"];')

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
    filename = "WGraphBessel3.svg"
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            pass
    if len(sys.argv) > 2:
        filename = sys.argv[2]

    wg = WGraphBessel3(n)
    wg.compute_cell()

    if filename.lower().endswith('.pdf'):
        wg.save_pdf(filename)
    else:
        wg.save_svg(filename)
