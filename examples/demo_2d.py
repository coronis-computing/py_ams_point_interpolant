"""Console script for py_ams_point_interpolant"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from py_ams_point_interpolant.ams_point_interpolant import ams_point_interpolant_2d


# ---------------------------------------------------------------------------
# Ground-truth function registry
# ---------------------------------------------------------------------------

FUNCTIONS = {}


def register(name, description, domain=(-np.pi, np.pi, -np.pi, np.pi)):
    """Decorator that registers a ground-truth function by name."""

    def decorator(fn):
        FUNCTIONS[name] = {"fn": fn, "description": description, "domain": domain}
        return fn

    return decorator


@register(
    "sin_cos",
    "sin(x)*cos(y)  —  smooth, periodic baseline",
)
def f_sin_cos(x, y):
    return np.sin(x) * np.cos(y)


@register(
    "ripple_gaussian",
    "Radial ripple with a Gaussian hill offset from the origin",
)
def f_ripple_gaussian(x, y):
    r = np.sqrt(x**2 + y**2)
    ripple = np.sin(r * 2.5) * np.exp(-r * 0.4)
    hill = np.exp(-((x - 0.5) ** 2 + (y + 0.5) ** 2) * 3)
    return ripple + hill


@register(
    "peaks",
    "MATLAB-style peaks function: a mix of Gaussians with positive and negative lobes",
)
def f_peaks(x, y):
    return (
        3 * (1 - x) ** 2 * np.exp(-(x**2) - (y + 1) ** 2)
        - 10 * (x / 5 - x**3 - y**5) * np.exp(-(x**2) - y**2)
        - (1 / 3) * np.exp(-((x + 1) ** 2) - y**2)
    )


@register(
    "franke",
    "Franke's function: standard scattered-data interpolation benchmark",
    domain=(0, 1, 0, 1),
)
def f_franke(x, y):
    t1 = 0.75 * np.exp(-((9 * x - 2) ** 2) / 4 - ((9 * y - 2) ** 2) / 4)
    t2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49 - (9 * y + 1) / 10)
    t3 = 0.5 * np.exp(-((9 * x - 7) ** 2) / 4 - ((9 * y - 3) ** 2) / 4)
    t4 = -0.2 * np.exp(-((9 * x - 4) ** 2) - ((9 * y - 7) ** 2))
    return t1 + t2 + t3 + t4


@register(
    "saddle_trig",
    "Saddle surface modulated by high-frequency trig: f = (x²-y²)*sin(3x)*cos(3y)",
)
def f_saddle_trig(x, y):
    return (x**2 - y**2) * np.sin(3 * x) * np.cos(3 * y)


@register(
    "sharp_ridge",
    "Smooth plateau crossed by a sharp exponential ridge along y = x",
)
def f_sharp_ridge(x, y):
    ridge = np.exp(-((x - y) ** 2) * 10)
    plateau = np.tanh(x + y)
    return ridge + 0.5 * plateau


@register(
    "checkerboard",
    "Smooth checkerboard: sign(sin(3x)*sin(3y)) blurred with a Gaussian envelope",
)
def f_checkerboard(x, y):
    envelope = np.exp(-(x**2 + y**2) * 0.15)
    return np.sin(3 * x) * np.sin(3 * y) * envelope


@register(
    "spiral",
    "Rotating spiral wave: sin(r² + 4θ) with radial decay",
)
def f_spiral(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return np.sin(r**2 + 4 * theta) * np.exp(-r * 0.3)


@register(
    "multi_scale",
    "Two superimposed sine waves at very different spatial scales",
)
def f_multi_scale(x, y):
    low = np.sin(x) * np.cos(y)
    high = 0.2 * np.sin(7 * x) * np.sin(7 * y)
    return low + high


@register(
    "cliff",
    "Near-discontinuous step: smooth tanh cliff along the diagonal",
)
def f_cliff(x, y):
    return np.tanh(10 * (x + y))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualise AMS point interpolation against various ground-truth functions."
    )
    parser.add_argument(
        "--function",
        default="sharp_ridge",
        choices=list(FUNCTIONS.keys()),
        metavar="NAME",
        help=(
            "Ground-truth function to use (default: sin_cos). "
            "Run with --list-functions to see all options."
        ),
    )
    parser.add_argument(
        "--list-functions",
        action="store_true",
        help="Print available ground-truth functions and exit.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of scattered samples (default: 10000)",
    )
    parser.add_argument(
        "--show_samples",
        action="store_true",
        help="Overlay sample points on the interpolation plot.",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=200,
        help="Evaluation grid resolution (default: 200)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --list-functions early exit
    if args.list_functions:
        print("\nAvailable ground-truth functions:\n")
        max_len = max(len(k) for k in FUNCTIONS)
        for name, meta in FUNCTIONS.items():
            domain = meta["domain"]
            dom_str = f"x∈[{domain[0]:.2g},{domain[1]:.2g}]  y∈[{domain[2]:.2g},{domain[3]:.2g}]"
            print(f"  {name:<{max_len}}   {meta['description']}")
            print(f"  {'':<{max_len}}   domain: {dom_str}\n")
        return 0

    # Retrieve selected function and its domain
    entry = FUNCTIONS[args.function]
    f = entry["fn"]
    x_lo, x_hi, y_lo, y_hi = entry["domain"]

    print(f"\n[function]  {args.function}: {entry['description']}")
    print(f"[domain]    x ∈ [{x_lo}, {x_hi}],  y ∈ [{y_lo}, {y_hi}]")
    print(f"[samples]   {args.num_samples}")
    print(f"[grid]      {args.grid_size}×{args.grid_size}\n")

    # Ground-truth grid
    n = args.grid_size
    x = np.linspace(x_lo, x_hi, n)
    y = np.linspace(y_lo, y_hi, n)
    X, Y = np.meshgrid(x, y)
    Z_true = f(X, Y)

    # Scattered sample points
    rng = np.random.default_rng()
    xs = rng.uniform(x_lo, x_hi, args.num_samples)
    ys = rng.uniform(y_lo, y_hi, args.num_samples)
    zs = f(xs, ys)
    points = np.column_stack((xs, ys))
    values = zs

    # Interpolate
    interp = ams_point_interpolant_2d(
        points, values, verbose=False, exact_interpolation=False, depth=8, parallel=1
    )
    eval_pts = np.column_stack((X.flatten(), Y.flatten()))
    Z_interp = interp.eval(eval_pts).reshape(n, n)

    # Error metrics
    abs_error = np.abs(Z_interp - Z_true)
    mae = np.mean(abs_error)
    rmse = np.sqrt(np.mean((Z_interp - Z_true) ** 2))
    max_err = np.max(abs_error)
    print(f"\nMAE      : {mae:.6f}")
    print(f"RMSE     : {rmse:.6f}")
    print(f"Max error: {max_err:.6f}")

    # Plot
    extent = [x_lo, x_hi, y_lo, y_hi]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        f"{args.function}  —  {entry['description']}\n"
        f"samples={args.num_samples}   MAE={mae:.4e}   RMSE={rmse:.4e}",
        fontsize=10,
    )

    vmin, vmax = Z_true.min(), Z_true.max()

    im0 = axs[0].imshow(Z_true, extent=extent, origin="lower", vmin=vmin, vmax=vmax)
    axs[0].set_title("Ground Truth")
    plt.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(Z_interp, extent=extent, origin="lower", vmin=vmin, vmax=vmax)
    axs[1].set_title("Interpolated")
    plt.colorbar(im1, ax=axs[1])
    if args.show_samples:
        axs[1].scatter(xs, ys, c="red", s=2, alpha=0.4, label="samples")
        axs[1].legend(loc="upper right", fontsize=7)

    im2 = axs[2].imshow(abs_error, extent=extent, origin="lower", cmap="hot")
    axs[2].set_title("Absolute Error")
    plt.colorbar(im2, ax=axs[2])

    plt.tight_layout()
    plt.show(block=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
