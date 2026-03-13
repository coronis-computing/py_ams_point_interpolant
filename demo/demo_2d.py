import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from py_ams_point_interpolant.ams_point_interpolant import (
    ams_point_interpolant_2d,
    get_parallel_types,
)

# ---------------------------------------------------------------------------
# Ground-truth function registry
# Each entry stores:
#   fn(x, y)    → scalar field values
#   grad(x, y)  → (dfdx, dfdy) as a tuple of arrays
#   description → human-readable string
#   domain      → (x_lo, x_hi, y_lo, y_hi)
#
# Decorated functions must return (value, (dfdx, dfdy)).
# ---------------------------------------------------------------------------

FUNCTIONS = {}
parallel_types = get_parallel_types()


def register(name, description, domain=(-np.pi, np.pi, -np.pi, np.pi)):
    def decorator(fn):
        FUNCTIONS[name] = {
            "fn": lambda x, y: fn(x, y)[0],
            "grad": lambda x, y: fn(x, y)[1],
            "description": description,
            "domain": domain,
        }
        return fn

    return decorator


@register("sin_cos", "sin(x)*cos(y)  —  smooth, periodic baseline")
def f_sin_cos(x, y):
    v = np.sin(x) * np.cos(y)
    return v, (np.cos(x) * np.cos(y), -np.sin(x) * np.sin(y))


@register(
    "ripple_gaussian", "Radial ripple with a Gaussian hill offset from the origin"
)
def f_ripple_gaussian(x, y):
    eps = 1e-12
    r = np.sqrt(x**2 + y**2)
    rip = np.sin(2.5 * r) * np.exp(-0.4 * r)
    drdr = (2.5 * np.cos(2.5 * r) - 0.4 * np.sin(2.5 * r)) * np.exp(-0.4 * r)
    hill = np.exp(-3 * ((x - 0.5) ** 2 + (y + 0.5) ** 2))
    v = rip + hill
    drdx = x / (r + eps)
    drdy = y / (r + eps)
    dfdx = drdr * drdx - 6 * (x - 0.5) * hill
    dfdy = drdr * drdy - 6 * (y + 0.5) * hill
    return v, (dfdx, dfdy)


@register(
    "peaks", "MATLAB-style peaks: mix of Gaussians with positive and negative lobes"
)
def f_peaks(x, y):
    e1 = np.exp(-(x**2) - (y + 1) ** 2)
    e2 = np.exp(-(x**2) - y**2)
    e3 = np.exp(-((x + 1) ** 2) - y**2)
    v = 3 * (1 - x) ** 2 * e1 - 10 * (x / 5 - x**3 - y**5) * e2 - (1 / 3) * e3
    dt1dx = 3 * ((-2 * (1 - x)) * e1 + (1 - x) ** 2 * (-2 * x) * e1)
    dt1dy = 3 * (1 - x) ** 2 * (-2 * (y + 1)) * e1
    dt2dx = -10 * ((1 / 5 - 3 * x**2) * e2 + (x / 5 - x**3 - y**5) * (-2 * x) * e2)
    dt2dy = -10 * (-5 * y**4 * e2 + (x / 5 - x**3 - y**5) * (-2 * y) * e2)
    dt3dx = -(1 / 3) * (-2 * (x + 1)) * e3
    dt3dy = -(1 / 3) * (-2 * y) * e3
    return v, (dt1dx + dt2dx + dt3dx, dt1dy + dt2dy + dt3dy)


@register(
    "franke",
    "Franke's function: standard scattered-data benchmark",
    domain=(0, 1, 0, 1),
)
def f_franke(x, y):
    e1 = np.exp(-((9 * x - 2) ** 2) / 4 - ((9 * y - 2) ** 2) / 4)
    e2 = np.exp(-((9 * x + 1) ** 2) / 49 - (9 * y + 1) / 10)
    e3 = np.exp(-((9 * x - 7) ** 2) / 4 - ((9 * y - 3) ** 2) / 4)
    e4 = np.exp(-((9 * x - 4) ** 2) - ((9 * y - 7) ** 2))
    v = 0.75 * e1 + 0.75 * e2 + 0.5 * e3 - 0.2 * e4
    dfdx = (
        0.75 * e1 * (-(9 / 2) * (9 * x - 2))
        + 0.75 * e2 * (-(18 / 49) * (9 * x + 1))
        + 0.5 * e3 * (-(9 / 2) * (9 * x - 7))
        - 0.2 * e4 * (-18 * (9 * x - 4))
    )
    dfdy = (
        0.75 * e1 * (-(9 / 2) * (9 * y - 2))
        + 0.75 * e2 * (-9 / 10)
        + 0.5 * e3 * (-(9 / 2) * (9 * y - 3))
        - 0.2 * e4 * (-18 * (9 * y - 7))
    )
    return v, (dfdx, dfdy)


@register("saddle_trig", "Saddle surface modulated by trig: (x²-y²)*sin(3x)*cos(3y)")
def f_saddle_trig(x, y):
    s3x, c3x = np.sin(3 * x), np.cos(3 * x)
    s3y, c3y = np.sin(3 * y), np.cos(3 * y)
    v = (x**2 - y**2) * s3x * c3y
    dfdx = 2 * x * s3x * c3y + (x**2 - y**2) * 3 * c3x * c3y
    dfdy = -2 * y * s3x * c3y + (x**2 - y**2) * s3x * (-3 * s3y)
    return v, (dfdx, dfdy)


@register(
    "sharp_ridge", "Smooth plateau crossed by a sharp exponential ridge along y = x"
)
def f_sharp_ridge(x, y):
    diff = x - y
    ridge = np.exp(-10 * diff**2)
    plat = np.tanh(x + y)
    v = ridge + 0.5 * plat
    dr = ridge * (-20 * diff)
    dp = 0.5 * (1 - plat**2)
    return v, (dr + dp, -dr + dp)


@register("checkerboard", "Smooth checkerboard: sin(3x)*sin(3y) with Gaussian envelope")
def f_checkerboard(x, y):
    env = np.exp(-(x**2 + y**2) * 0.15)
    s3x, c3x = np.sin(3 * x), np.cos(3 * x)
    s3y, c3y = np.sin(3 * y), np.cos(3 * y)
    v = s3x * s3y * env
    dfdx = (3 * c3x * s3y - 0.3 * x * s3x * s3y) * env
    dfdy = (s3x * 3 * c3y - 0.3 * y * s3x * s3y) * env
    return v, (dfdx, dfdy)


@register("spiral", "Rotating spiral wave: sin(r² + 4θ) with radial decay")
def f_spiral(x, y):
    eps = 1e-12
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    phi = r**2 + 4 * theta
    decay = np.exp(-0.3 * r)
    v = np.sin(phi) * decay
    r2 = r**2 + eps
    dphidx = 2 * x - 4 * y / r2
    dphidy = 2 * y + 4 * x / r2
    ddx = -0.3 * x / (r + eps) * decay
    ddy = -0.3 * y / (r + eps) * decay
    cp = np.cos(phi)
    dfdx = cp * decay * dphidx + np.sin(phi) * ddx
    dfdy = cp * decay * dphidy + np.sin(phi) * ddy
    return v, (dfdx, dfdy)


@register("multi_scale", "Low + high frequency superimposed sine waves")
def f_multi_scale(x, y):
    v = np.sin(x) * np.cos(y) + 0.2 * np.sin(7 * x) * np.sin(7 * y)
    dfdx = np.cos(x) * np.cos(y) + 1.4 * np.cos(7 * x) * np.sin(7 * y)
    dfdy = -np.sin(x) * np.sin(y) + 1.4 * np.sin(7 * x) * np.cos(7 * y)
    return v, (dfdx, dfdy)


@register("cliff", "Near-discontinuous tanh cliff along the diagonal")
def f_cliff(x, y):
    t = np.tanh(10 * (x + y))
    dt = 10 * (1 - t**2)
    return t, (dt, dt)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualise AMS 2-D point interpolation with optional gradient data."
    )
    parser.add_argument(
        "--function",
        default="sin_cos",
        choices=list(FUNCTIONS.keys()),
        metavar="NAME",
        help="Ground-truth function (default: sin_cos). Use --list-functions to see all.",
    )
    parser.add_argument(
        "--list-functions",
        action="store_true",
        help="Print available functions and exit.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of scattered samples (default: 10000)",
    )
    parser.add_argument(
        "--use_gradients",
        action="store_true",
        help="Pass analytical gradients to the interpolant.",
    )
    parser.add_argument(
        "--grad_fraction",
        type=float,
        default=1.0,
        help="Fraction of sample points that supply a gradient "
        "when --use_gradients is set (default: 1.0 = all)",
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
    parser.add_argument(
        "--depth",
        type=int,
        default=8,
        help="This integer is the maximum depth of the tree that will be used for surface reconstruction. Running at depth d corresponds to solving on a grid whose resolution is no larger than 2^d x 2^d x ... Note that since the reconstructor adapts the octree to the sampling density, the specified reconstruction depth is only an upper bound. (default: 8)",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=2,
        help="Degree of the B-spline that is to be used to define the finite elements system. Larger degrees support higher order approximations, but come at the cost of denser system matrices (incurring a cost in both space and time). (default: 2)",
    )
    parser.add_argument(
        "--solve_depth",
        type=int,
        default=-1,
        help=" the depth up to which the solver will solve the numerical system. It will still show the results at the finest resolution, but no additional high-frequency data will be introduced at the finest resolutions. It could also be the case that aliasing that occurs at the coarser resolutions will not get corrected. (default = -1, i.e., --depth)",
    )
    parser.add_argument(
        "--full_depth",
        type=int,
        default=5,
        help="The depth up to which the octree is completely refined, i.e. a regular grid (default: 5)",
    )
    parser.add_argument(
        "--base_depth",
        type=int,
        default=-1,
        help="The coarsest depth at which the system will be solved over an octree. (At coarser levels it will be solved using a standard MG solver, with multiple V-Cycles, defined over a regular grid.) As such, the assumption is that BaseDepth<=FullDepth (default = -1, i.e., not used)",
    )
    parser.add_argument(
        "--boundary_type",
        type=str,
        default="free",
        help="Boundary type (default: free, available: free, dirichlet, neumann)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=8,
        help="The number of Gauss-Seidel relaxations to be performed at every level of the hierarchy (default: 8)",
    )
    parser.add_argument(
        "--base_v_cycles",
        type=int,
        default=4,
        help="coarse MG solver v-cycles (default: 4)",
    )
    parser.add_argument(
        "--max_memory_gb",
        type=int,
        default=0,
        help="Maximum memory to use in GB (default: 0, i.e., no limit)",
    )
    parser.add_argument(
        "--parallel_type",
        default="openmp",
        help="Parallel mode (default: openmp, available: "
        + ", ".join(parallel_types)
        + " )",
    )
    parser.add_argument(
        "--parallel_schedule",
        type=str,
        default="static",
        help="Parallel schedule (default: static, available: static, dynamic)",
    )
    parser.add_argument(
        "--parallel_thread_chunk_size",
        type=int,
        default=128,
        help="Parallel thread chunk size (default: 128)",
    )
    parser.add_argument(
        "--value_weight",
        type=float,
        default=1000.0,
        help="Importance that interpolation of the samples' values is given in the fitting of the function (default: 1000.0)",
    )
    parser.add_argument(
        "--gradient_weight",
        type=float,
        default=1.0,
        help="Importance that interpolation of the samples' gradients is given in the fitting of the function (default: 1.0)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.1,
        help="The ratio between the diameter of the cube used for reconstruction and the diameter of the samples' bounding cube. (default: 1.1)",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=0.0,
        help="Target width of the finest level octree cells. This parameter is ignored if the --depth is also specified. (default: 0.0, i.e., ignore and use --depth)",
    )
    parser.add_argument(
        "--cg_accuracy",
        type=float,
        default=1e-3,
        help="Conjugate Gradient solver accuracy (default: 1e-3)",
    )
    parser.add_argument(
        "--iso", type=float, default=0.0, help="Iso-value (default=0.0)"
    )
    parser.add_argument(
        "--lap_weight",
        type=float,
        default=0.0,
        help="Importance that Laplacian regularization is given in the fitting of the function (default: 0.0)",
    )
    parser.add_argument(
        "--bi_lap_weight",
        type=float,
        default=1.0,
        help="Importance that bi-Laplacian regularization is given in the fitting of the function (default: 1.0)",
    )
    parser.add_argument(
        "--show_performance",
        action="store_true",
        help="Show performance statistics (default: false)",
    )
    parser.add_argument(
        "--show_residual", action="store_true", help="Show residuals (default: false)"
    )
    parser.add_argument(
        "--exact_interpolation",
        action="store_true",
        help="Use exact interpolation (default: false)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose mode (only affects the creation of the interpolant, the demo will print information on screen anyway) (default: false)",
    )
    parser.add_argument(
        "--transform_file", type=str, default="", help="Transform file (default: none)"
    )
    return parser.parse_args()


def validate_args(args):
    if args.parallel_type not in parallel_types:
        raise ValueError(
            f"Invalid parallel type: {args.parallel_type}, available: "
            + " ".join(parallel_types)
        )

    if args.parallel_schedule not in ["static", "dynamic"]:
        raise ValueError(
            f"Invalid parallel schedule: {args.parallel_schedule}, available: static, dynamic"
        )

    if args.boundary_type not in ["free", "dirichlet", "neumann"]:
        raise ValueError(
            f"Invalid boundary type: {args.boundary_type}, available: free, dirichlet, neumann"
        )

    if args.degree < 2 or args.degree > 3:
        raise ValueError(f"Invalid degree: {args.degree}, available: 2, 3")


def main():
    args = parse_args()
    validate_args(args)

    if args.list_functions:
        print("\nAvailable 2-D ground-truth functions:\n")
        max_len = max(len(k) for k in FUNCTIONS)
        for name, meta in FUNCTIONS.items():
            d = meta["domain"]
            print(f"  {name:<{max_len}}   {meta['description']}")
            print(
                f"  {'':<{max_len}}   domain: x∈[{d[0]:.2g},{d[1]:.2g}]  "
                f"y∈[{d[2]:.2g},{d[3]:.2g}]\n"
            )
        return 0

    entry = FUNCTIONS[args.function]
    f = entry["fn"]
    grad_f = entry["grad"]
    x_lo, x_hi, y_lo, y_hi = entry["domain"]

    grad_tag = (
        f"  +gradients ({int(args.grad_fraction * 100)} % of points)"
        if args.use_gradients
        else "  (no gradients)"
    )
    print(f"\n[function]  {args.function}: {entry['description']}")
    print(f"[domain]    x∈[{x_lo},{x_hi}]  y∈[{y_lo},{y_hi}]")
    print(f"[samples]   {args.num_samples}{grad_tag}")
    print(f"[grid]      {args.grid_size}×{args.grid_size}")

    # Ground-truth grid
    n = args.grid_size
    xg = np.linspace(x_lo, x_hi, n)
    yg = np.linspace(y_lo, y_hi, n)
    X, Y = np.meshgrid(xg, yg)
    Z_true = f(X, Y)

    # Scattered sample points
    rng = np.random.default_rng()
    xs = rng.uniform(x_lo, x_hi, args.num_samples)
    ys = rng.uniform(y_lo, y_hi, args.num_samples)
    vals = f(xs, ys)
    points = np.column_stack((xs, ys))

    # Gradients
    if args.use_gradients:
        n_grad = max(1, int(args.num_samples * args.grad_fraction))
        grad_idx = rng.choice(args.num_samples, size=n_grad, replace=False)
        grad_pts = points[grad_idx]  # (n_grad, 2)
        gx, gy = grad_f(xs[grad_idx], ys[grad_idx])
        gradients = np.column_stack((gx, gy))  # (n_grad, 2)
        print(f"[gradients] providing {n_grad} gradient vectors")
    else:
        grad_pts = np.array([])
        gradients = np.array([])

    # Build the interpolant
    interp = ams_point_interpolant_2d(
        points,
        vals,
        grad_pts,
        gradients,
        depth=args.depth,
        degree=args.degree,
        solve_depth=args.solve_depth,
        full_depth=args.full_depth,
        base_depth=args.base_depth,
        b_type=["free", "neumann", "dirichlet"].index(args.boundary_type) + 1,
        iters=args.iters,
        base_v_cycles=args.base_v_cycles,
        max_memory_gb=args.max_memory_gb,
        parallel=parallel_types.index(args.parallel_type),
        schedule=["static", "dynamic"].index(args.parallel_schedule),
        thread_chunk_size=args.parallel_thread_chunk_size,
        value_weight=args.value_weight,
        gradient_weight=args.gradient_weight,
        scale=args.scale,
        width=args.width,
        cg_accuracy=args.cg_accuracy,
        iso=args.iso,
        lap_weight=args.lap_weight,
        bi_lap_weight=args.bi_lap_weight,
        show_performance=args.show_performance,
        show_residual=args.show_residual,
        exact_interpolation=args.exact_interpolation,
        verbose=args.verbose,
        transform_file=args.transform_file,
    )

    # Evaluate on grid
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
        f"{args.function}{grad_tag}  —  {entry['description']}\n"
        f"samples={args.num_samples}   MAE={mae:.3e}   RMSE={rmse:.3e}   max={max_err:.3e}",
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
        axs[1].scatter(xs, ys, c="red", s=2, alpha=0.3, linewidths=0, label="value pts")
        if args.use_gradients:
            axs[1].scatter(
                grad_pts[:, 0],
                grad_pts[:, 1],
                c="cyan",
                s=4,
                alpha=0.5,
                marker="x",
                label="grad pts",
            )
        axs[1].legend(loc="upper right", fontsize=7)

    im2 = axs[2].imshow(abs_error, extent=extent, origin="lower", cmap="hot")
    axs[2].set_title("Absolute Error")
    plt.colorbar(im2, ax=axs[2])

    plt.tight_layout()
    plt.show(block=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
