"""Console script for py_ams_point_interpolant — 3-D version"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from py_ams_point_interpolant.ams_point_interpolant import ams_point_interpolant_3d


# ---------------------------------------------------------------------------
# Ground-truth function registry
# ---------------------------------------------------------------------------

FUNCTIONS = {}


def register(name, description, domain=(-np.pi, np.pi, -np.pi, np.pi, -np.pi, np.pi)):
    """Decorator that registers a 3-D ground-truth function by name."""

    def decorator(fn):
        FUNCTIONS[name] = {"fn": fn, "description": description, "domain": domain}
        return fn

    return decorator


@register(
    "sin_cos_sin",
    "sin(x)*cos(y)*sin(z)  —  smooth, triply-periodic baseline",
)
def f_sin_cos_sin(x, y, z):
    return np.sin(x) * np.cos(y) * np.sin(z)


@register(
    "ripple_gaussian_3d",
    "Radial 3-D ripple damped by a Gaussian: sin(r*2.5)*exp(-r*0.4)",
)
def f_ripple_gaussian_3d(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    return np.sin(r * 2.5) * np.exp(-r * 0.4)


@register(
    "franke_3d",
    "3-D extension of Franke's benchmark: sum of Gaussians with mixed signs",
    domain=(0, 1, 0, 1, 0, 1),
)
def f_franke_3d(x, y, z):
    t1 = 0.75 * np.exp(
        -((9 * x - 2) ** 2) / 4 - ((9 * y - 2) ** 2) / 4 - ((9 * z - 2) ** 2) / 4
    )
    t2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49 - (9 * y + 1) / 10 - (9 * z + 1) / 10)
    t3 = 0.50 * np.exp(
        -((9 * x - 7) ** 2) / 4 - ((9 * y - 3) ** 2) / 4 - ((9 * z - 5) ** 2) / 4
    )
    t4 = -0.20 * np.exp(-((9 * x - 4) ** 2) - ((9 * y - 7) ** 2) - ((9 * z - 5) ** 2))
    return t1 + t2 + t3 + t4


@register(
    "multi_scale_3d",
    "Low + high frequency superimposed: sin(x)cos(y)sin(z) + 0.2*sin(6x)sin(6y)sin(6z)",
)
def f_multi_scale_3d(x, y, z):
    low = np.sin(x) * np.cos(y) * np.sin(z)
    high = 0.2 * np.sin(6 * x) * np.sin(6 * y) * np.sin(6 * z)
    return low + high


@register(
    "sharp_blob",
    "Three narrow Gaussian blobs at different corners of the domain",
)
def f_sharp_blob(x, y, z):
    def blob(cx, cy, cz, s=4.0):
        return np.exp(-s * ((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2))

    return blob(1.5, 1.5, 1.5) - blob(-1.5, -1.5, -1.5) + 0.5 * blob(1.5, -1.5, 0.0)


@register(
    "saddle_trig_3d",
    "3-D saddle modulated by trig: (x²-y²+0.5*z²)*sin(2x)*cos(2y)*sin(2z)",
)
def f_saddle_trig_3d(x, y, z):
    return (x**2 - y**2 + 0.5 * z**2) * np.sin(2 * x) * np.cos(2 * y) * np.sin(2 * z)


@register(
    "cliff_3d",
    "Near-discontinuous tanh cliff along the plane x+y+z=0",
)
def f_cliff_3d(x, y, z):
    return np.tanh(10 * (x + y + z) / np.sqrt(3))


@register(
    "spiral_tube",
    "Helicoidal wave: sin(r²_xy + 4θ + 2z) with radial + axial decay",
)
def f_spiral_tube(x, y, z):
    r_xy = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return (
        np.sin(r_xy**2 + 4 * theta + 2 * z)
        * np.exp(-r_xy * 0.3)
        * np.exp(-(z**2) * 0.1)
    )


@register(
    "trig_product",
    "cos(x)*sin(y+z) + sin(x*y)*cos(z)  —  non-separable cross-terms",
)
def f_trig_product(x, y, z):
    return np.cos(x) * np.sin(y + z) + np.sin(x * y) * np.cos(z)


@register(
    "radial_shell",
    "Thin Gaussian shell at r=π/2: exp(-20*(r - π/2)²)",
)
def f_radial_shell(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    return np.exp(-20 * (r - np.pi / 2) ** 2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualise AMS 3-D point interpolation against various ground-truth functions.\n"
            "Results are shown as three orthogonal cross-sections through the domain centre."
        )
    )
    parser.add_argument(
        "--function",
        default="sin_cos_sin",
        choices=list(FUNCTIONS.keys()),
        metavar="NAME",
        help="Ground-truth function to use (default: sin_cos_sin). Use --list-functions to see all.",
    )
    parser.add_argument(
        "--list-functions",
        action="store_true",
        help="Print available ground-truth functions and exit.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50000,
        help="Number of scattered 3-D samples (default: 50000)",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=64,
        help="Evaluation grid resolution per axis (default: 64); total eval pts = grid_size³",
    )
    parser.add_argument(
        "--slice_axis",
        type=str,
        default="all",
        choices=["x", "y", "z", "all"],
        help="Which axis to slice through the domain centre for visualisation (default: all)",
    )
    parser.add_argument(
        "--show_samples",
        action="store_true",
        help="Overlay projected sample points on the slice plots.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_grid(domain, n):
    """Return (x1d, y1d, z1d, X, Y, Z) for an n³ meshgrid over domain."""
    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = domain
    x1 = np.linspace(x_lo, x_hi, n)
    y1 = np.linspace(y_lo, y_hi, n)
    z1 = np.linspace(z_lo, z_hi, n)
    X, Y, Z = np.meshgrid(x1, y1, z1, indexing="ij")
    return x1, y1, z1, X, Y, Z


def centre_slice(arr3d, axis, idx=None):
    """Extract the 2-D mid-plane of a 3-D array along the given axis (0,1,2)."""
    n = arr3d.shape[axis]
    i = n // 2 if idx is None else idx
    if axis == 0:
        return arr3d[i, :, :]
    elif axis == 1:
        return arr3d[:, i, :]
    else:
        return arr3d[:, :, i]


def add_slice_panel(
    fig,
    axs_row,
    Z_true_3d,
    Z_interp_3d,
    axis,
    ax_labels,
    domain,
    samples_xy=None,
    show_samples=False,
):
    """
    Fill one row of 3 axes: ground truth | interpolated | absolute error
    for a single orthogonal slice.
    """
    slice_true = centre_slice(Z_true_3d, axis)
    slice_interp = centre_slice(Z_interp_3d, axis)
    abs_err = np.abs(slice_interp - slice_true)

    # Map axis index → extent order [h_lo, h_hi, v_lo, v_hi]
    d = domain  # (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
    axis_extents = {
        0: [d[2], d[3], d[4], d[5]],  # slice ⊥ x  →  YZ plane
        1: [d[0], d[1], d[4], d[5]],  # slice ⊥ y  →  XZ plane
        2: [d[0], d[1], d[2], d[3]],  # slice ⊥ z  →  XY plane
    }
    extent = axis_extents[axis]
    h_label, v_label = ax_labels

    vmin, vmax = slice_true.min(), slice_true.max()

    titles = ["Ground Truth", "Interpolated", "Absolute Error"]
    datas = [slice_true, slice_interp, abs_err]
    cmaps = ["viridis", "viridis", "hot"]
    vmins = [vmin, vmin, None]
    vmaxs = [vmax, vmax, None]

    for ax, title, data, cmap, v0, v1 in zip(
        axs_row, titles, datas, cmaps, vmins, vmaxs
    ):
        kw = dict(extent=extent, origin="lower", aspect="auto", cmap=cmap)
        if v0 is not None:
            kw["vmin"] = v0
            kw["vmax"] = v1
        im = ax.imshow(data.T, **kw)
        ax.set_xlabel(h_label)
        ax.set_ylabel(v_label)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        if show_samples and samples_xy is not None:
            sh, sv = samples_xy
            ax.scatter(sh, sv, c="red", s=1, alpha=0.3, linewidths=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    # --list-functions early exit
    if args.list_functions:
        print("\nAvailable 3-D ground-truth functions:\n")
        max_len = max(len(k) for k in FUNCTIONS)
        for name, meta in FUNCTIONS.items():
            d = meta["domain"]
            dom_str = (
                f"x∈[{d[0]:.2g},{d[1]:.2g}]  "
                f"y∈[{d[2]:.2g},{d[3]:.2g}]  "
                f"z∈[{d[4]:.2g},{d[5]:.2g}]"
            )
            print(f"  {name:<{max_len}}   {meta['description']}")
            print(f"  {'':<{max_len}}   domain: {dom_str}\n")
        return 0

    entry = FUNCTIONS[args.function]
    f = entry["fn"]
    domain = entry["domain"]
    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = domain
    n = args.grid_size

    print(f"\n[function]  {args.function}: {entry['description']}")
    print(f"[domain]    x∈[{x_lo},{x_hi}]  y∈[{y_lo},{y_hi}]  z∈[{z_lo},{z_hi}]")
    print(f"[samples]   {args.num_samples}")
    print(f"[grid]      {n}³ = {n**3:,} eval points\n")

    # Ground-truth grid
    x1, y1, z1, X, Y, Z = make_grid(domain, n)
    Z_true_3d = f(X, Y, Z)  # shape (n, n, n)

    # Scattered samples
    rng = np.random.default_rng()
    xs = rng.uniform(x_lo, x_hi, args.num_samples)
    ys = rng.uniform(y_lo, y_hi, args.num_samples)
    zs_coord = rng.uniform(z_lo, z_hi, args.num_samples)
    vals = f(xs, ys, zs_coord)
    points = np.column_stack((xs, ys, zs_coord))

    # Interpolate
    interp = ams_point_interpolant_3d(
        points, vals, verbose=False, exact_interpolation=False, depth=8, parallel=1
    )
    eval_pts = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
    Z_interp_3d = interp.eval(eval_pts).reshape(n, n, n)

    # Error metrics (full 3-D volume)
    abs_error = np.abs(Z_interp_3d - Z_true_3d)
    mae = np.mean(abs_error)
    rmse = np.sqrt(np.mean((Z_interp_3d - Z_true_3d) ** 2))
    max_err = np.max(abs_error)
    print(f"\nMAE      : {mae:.6f}")
    print(f"RMSE     : {rmse:.6f}")
    print(f"Max error: {max_err:.6f}")

    # Determine which slices to show
    slice_map = {
        "z": [(2, ("x", "y"), (xs, ys))],
        "y": [(1, ("x", "z"), (xs, zs_coord))],
        "x": [(0, ("y", "z"), (ys, zs_coord))],
    }
    if args.slice_axis == "all":
        slices = [
            (2, ("x", "y"), (xs, ys)),
            (1, ("x", "z"), (xs, zs_coord)),
            (0, ("y", "z"), (ys, zs_coord)),
        ]
    else:
        slices = slice_map[args.slice_axis]

    n_rows = len(slices)
    fig, axs = plt.subplots(n_rows, 3, figsize=(15, 4.5 * n_rows))
    if n_rows == 1:
        axs = axs[np.newaxis, :]  # keep 2-D indexing consistent

    axis_names = {0: "YZ", 1: "XZ", 2: "XY"}
    fig.suptitle(
        f"{args.function}  —  {entry['description']}\n"
        f"samples={args.num_samples}   grid={n}³   "
        f"MAE={mae:.3e}   RMSE={rmse:.3e}   max={max_err:.3e}",
        fontsize=10,
    )

    for row_idx, (axis, (h_lbl, v_lbl), samp_xy) in enumerate(slices):
        mid_val = [x1, y1, z1][axis][n // 2]
        axs[row_idx, 0].set_ylabel(
            f"{axis_names[axis]} slice\n@ {'xyz'[axis]}={mid_val:.2f}\n{v_lbl}",
            fontsize=8,
        )
        add_slice_panel(
            fig,
            axs[row_idx],
            Z_true_3d,
            Z_interp_3d,
            axis=axis,
            ax_labels=(h_lbl, v_lbl),
            domain=domain,
            samples_xy=samp_xy if args.show_samples else None,
            show_samples=args.show_samples,
        )

    plt.tight_layout()
    plt.show(block=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
