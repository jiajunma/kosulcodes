import argparse
from pathlib import Path

from PP_to_PM import normalize_subset, parse_csv_ints


def column_caps_from_mu(m, n, I, J):
    """
    For mu=(I,J), compute cap[lambda] (1 <= lambda <= m):
      cap(lambda)=0 if lambda < i1,
      cap(lambda)=j_p if i_p <= lambda < i_{p+1},
      cap(lambda)=j_k for lambda >= i_k.
    """
    I = normalize_subset(I, m)
    J = normalize_subset(J, n)
    if len(I) != len(J):
        raise ValueError(f"Require |I|=|J|, got |I|={len(I)}, |J|={len(J)}")

    caps = [0] * m
    if not I:
        return caps

    p = 0
    for lam in range(1, m + 1):
        while p + 1 < len(I) and lam >= I[p + 1]:
            p += 1
        if lam >= I[0]:
            caps[lam - 1] = J[p]
    return caps


def _svg_escape(s):
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _svg_rect(x, y, w, h, fill="none", stroke="none", stroke_width=1, opacity=1.0):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" opacity="{opacity}"/>'
    )


def _svg_line(x1, y1, x2, y2, color="#000", width=1):
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{color}" stroke-width="{width}"/>'
    )


def _svg_circle(cx, cy, r, fill="#222", stroke="#222", stroke_width=1):
    return (
        f'<circle cx="{cx}" cy="{cy}" r="{r}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
    )


def _svg_text(x, y, text, size=12, color="#111", anchor="start"):
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
        f'text-anchor="{anchor}" font-family="Times New Roman, serif">{_svg_escape(text)}</text>'
    )


def draw_block_configuration(m, n, I, J, output_path):
    I = normalize_subset(I, m)
    J = normalize_subset(J, n)
    caps = column_caps_from_mu(m, n, I, J)

    cell = 44
    left = 90
    top = 90
    width = left + m * cell + 80
    height = top + n * cell + 110

    def x_of_col(col):
        return left + (col - 1) * cell

    def y_of_row(row):
        return top + (row - 1) * cell

    lines = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    )
    lines.append(_svg_rect(0, 0, width, height, fill="white"))

    # Title
    lines.append(
        _svg_text(
            width / 2,
            36,
            f"Block configuration of L1^mu, mu=(I={tuple(I)}, J={tuple(J)})",
            size=19,
            anchor="middle",
        )
    )

    # Shade allowed cells.
    for lam in range(1, m + 1):
        cap = caps[lam - 1]
        for row in range(1, cap + 1):
            lines.append(
                _svg_rect(
                    x_of_col(lam),
                    y_of_row(row),
                    cell,
                    cell,
                    fill="#d9d9d9",
                    stroke="none",
                )
            )

    # Grid lines.
    for c in range(0, m + 1):
        x = left + c * cell
        lines.append(_svg_line(x, top, x, top + n * cell, color="#cfcfcf", width=1))
    for r in range(0, n + 1):
        y = top + r * cell
        lines.append(_svg_line(left, y, left + m * cell, y, color="#cfcfcf", width=1))

    # Stair boundary.
    for lam in range(1, m + 1):
        cap = caps[lam - 1]
        y = top + cap * cell
        x1 = x_of_col(lam)
        x2 = x1 + cell
        lines.append(_svg_line(x1, y, x2, y, color="#222", width=3))
        if lam < m:
            y2 = top + caps[lam] * cell
            lines.append(_svg_line(x2, y, x2, y2, color="#222", width=3))

    # Corners (j_p, i_p).
    for ip, jp in zip(I, J):
        cx = x_of_col(ip) + cell / 2
        cy = y_of_row(jp) + cell / 2
        lines.append(_svg_circle(cx, cy, 5.5, fill="#222"))
        lines.append(_svg_text(cx + 10, cy - 6, f"({jp},{ip})", size=14))

    # Axis ticks.
    for lam in range(1, m + 1):
        x = x_of_col(lam) + cell / 2
        lines.append(_svg_text(x, top + n * cell + 26, str(lam), size=13, anchor="middle"))
    for row in range(1, n + 1):
        y = y_of_row(row) + cell / 2 + 4
        lines.append(_svg_text(left - 14, y, str(row), size=13, anchor="end"))

    # Axis labels.
    lines.append(
        _svg_text(left + (m * cell) / 2, top + n * cell + 58, "column λ (e_λ)", size=14, anchor="middle")
    )
    lines.append(_svg_text(16, top + (n * cell) / 2, "row j", size=14))

    lines.append("</svg>")

    output_path = Path(output_path)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Draw block matrix configuration for the linear subspace \\bar L_1^mu."
    )
    parser.add_argument("m", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("--I", required=True, help='CSV subset, e.g. "3,5,7"')
    parser.add_argument("--J", required=True, help='CSV subset, e.g. "2,4,6"')
    parser.add_argument("--output", default="block_configuration.svg")
    args = parser.parse_args()

    if args.m <= 0 or args.n <= 0:
        raise ValueError(f"Require m,n >= 1, got m={args.m}, n={args.n}")

    I = parse_csv_ints(args.I)
    J = parse_csv_ints(args.J)
    path = draw_block_configuration(args.m, args.n, I, J, args.output)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
