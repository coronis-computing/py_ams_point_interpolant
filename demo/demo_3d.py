import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from py_ams_point_interpolant.ams_point_interpolant import (
    ams_point_interpolant_3d,
    get_parallel_types,
)


# ---------------------------------------------------------------------------
# Ground-truth function registry
# Each entry stores:
#   fn(x,y,z)   → scalar field values
#   grad(x,y,z) → (dfdx, dfdy, dfdz) as a tuple of arrays
#   description → human-readable string
#   domain      → (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
#
# Decorated functions must return (value, (dfdx, dfdy, dfdz)).
# ---------------------------------------------------------------------------

FUNCTIONS = {}
parallel_types = get_parallel_types()


def register(name, description, domain=(-np.pi, np.pi, -np.pi, np.pi, -np.pi, np.pi)):
    def decorator(fn):
        FUNCTIONS[name] = {
            "fn": lambda x, y, z: fn(x, y, z)[0],
            "grad": lambda x, y, z: fn(x, y, z)[1],
            "description": description,
            "domain": domain,
        }
        return fn

    return decorator


@register("sin_cos_sin", "sin(x)*cos(y)*sin(z)  —  smooth, triply-periodic baseline")
def f_sin_cos_sin(x, y, z):
    v = np.sin(x) * np.cos(y) * np.sin(z)
    dfdx = np.cos(x) * np.cos(y) * np.sin(z)
    dfdy = -np.sin(x) * np.sin(y) * np.sin(z)
    dfdz = np.sin(x) * np.cos(y) * np.cos(z)
    return v, (dfdx, dfdy, dfdz)


@register(
    "ripple_gaussian_3d", "Radial 3-D ripple with Gaussian decay: sin(2.5r)*exp(-0.4r)"
)
def f_ripple_gaussian_3d(x, y, z):
    eps = 1e-12
    r = np.sqrt(x**2 + y**2 + z**2)
    v = np.sin(2.5 * r) * np.exp(-0.4 * r)
    drdr = (2.5 * np.cos(2.5 * r) - 0.4 * np.sin(2.5 * r)) * np.exp(-0.4 * r)
    drdx, drdy, drdz = x / (r + eps), y / (r + eps), z / (r + eps)
    return v, (drdr * drdx, drdr * drdy, drdr * drdz)


@register("franke_3d", "3-D extension of Franke's benchmark", domain=(0, 1, 0, 1, 0, 1))
def f_franke_3d(x, y, z):
    e1 = np.exp(
        -((9 * x - 2) ** 2) / 4 - ((9 * y - 2) ** 2) / 4 - ((9 * z - 2) ** 2) / 4
    )
    e2 = np.exp(-((9 * x + 1) ** 2) / 49 - (9 * y + 1) / 10 - (9 * z + 1) / 10)
    e3 = np.exp(
        -((9 * x - 7) ** 2) / 4 - ((9 * y - 3) ** 2) / 4 - ((9 * z - 5) ** 2) / 4
    )
    e4 = np.exp(-((9 * x - 4) ** 2) - ((9 * y - 7) ** 2) - ((9 * z - 5) ** 2))
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
    dfdz = (
        0.75 * e1 * (-(9 / 2) * (9 * z - 2))
        + 0.75 * e2 * (-9 / 10)
        + 0.5 * e3 * (-(9 / 2) * (9 * z - 5))
        - 0.2 * e4 * (-18 * (9 * z - 5))
    )
    return v, (dfdx, dfdy, dfdz)


@register(
    "multi_scale_3d",
    "Low + high frequency: sin(x)cos(y)sin(z) + 0.2*sin(6x)sin(6y)sin(6z)",
)
def f_multi_scale_3d(x, y, z):
    lo = np.sin(x) * np.cos(y) * np.sin(z)
    hi = 0.2 * np.sin(6 * x) * np.sin(6 * y) * np.sin(6 * z)
    v = lo + hi
    dfdx = np.cos(x) * np.cos(y) * np.sin(z) + 1.2 * np.cos(6 * x) * np.sin(
        6 * y
    ) * np.sin(6 * z)
    dfdy = -np.sin(x) * np.sin(y) * np.sin(z) + 1.2 * np.sin(6 * x) * np.cos(
        6 * y
    ) * np.sin(6 * z)
    dfdz = np.sin(x) * np.cos(y) * np.cos(z) + 1.2 * np.sin(6 * x) * np.sin(
        6 * y
    ) * np.cos(6 * z)
    return v, (dfdx, dfdy, dfdz)


@register(
    "sharp_blob", "Three narrow Gaussian blobs at different corners of the domain"
)
def f_sharp_blob(x, y, z):
    def blob(cx, cy, cz, s=4.0):
        return np.exp(-s * ((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2))

    def dblob(cx, cy, cz, s=4.0):
        b = blob(cx, cy, cz, s)
        return b * (-2 * s * (x - cx)), b * (-2 * s * (y - cy)), b * (-2 * s * (z - cz))

    b1, b2, b3 = blob(1.5, 1.5, 1.5), blob(-1.5, -1.5, -1.5), blob(1.5, -1.5, 0.0)
    v = b1 - b2 + 0.5 * b3
    db1x, db1y, db1z = dblob(1.5, 1.5, 1.5)
    db2x, db2y, db2z = dblob(-1.5, -1.5, -1.5)
    db3x, db3y, db3z = dblob(1.5, -1.5, 0.0)
    return v, (
        db1x - db2x + 0.5 * db3x,
        db1y - db2y + 0.5 * db3y,
        db1z - db2z + 0.5 * db3z,
    )


@register("saddle_trig_3d", "(x²-y²+0.5z²)*sin(2x)*cos(2y)*sin(2z)")
def f_saddle_trig_3d(x, y, z):
    P = x**2 - y**2 + 0.5 * z**2
    T = np.sin(2 * x) * np.cos(2 * y) * np.sin(2 * z)
    v = P * T
    dTdx = 2 * np.cos(2 * x) * np.cos(2 * y) * np.sin(2 * z)
    dTdy = -2 * np.sin(2 * x) * np.sin(2 * y) * np.sin(2 * z)
    dTdz = 2 * np.sin(2 * x) * np.cos(2 * y) * np.cos(2 * z)
    dfdx = 2 * x * T + P * dTdx
    dfdy = -2 * y * T + P * dTdy
    dfdz = z * T + P * dTdz
    return v, (dfdx, dfdy, dfdz)


@register("cliff_3d", "Near-discontinuous tanh cliff along the plane x+y+z=0")
def f_cliff_3d(x, y, z):
    s = (x + y + z) / np.sqrt(3)
    t = np.tanh(10 * s)
    dt = 10 * (1 - t**2) / np.sqrt(3)
    return t, (dt, dt, dt)


@register(
    "spiral_tube", "Helicoidal wave: sin(r²_xy + 4θ + 2z)*exp(-0.3*r_xy)*exp(-0.1z²)"
)
def f_spiral_tube(x, y, z):
    eps = 1e-12
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    phi = r**2 + 4 * theta + 2 * z
    dec_r = np.exp(-0.3 * r)
    dec_z = np.exp(-0.1 * z**2)
    decay = dec_r * dec_z
    v = np.sin(phi) * decay
    r2 = r**2 + eps
    dphidx = 2 * x - 4 * y / r2
    dphidy = 2 * y + 4 * x / r2
    dphidz = 2.0
    cp = np.cos(phi)
    sp = np.sin(phi)
    ddrdx = -0.3 * x / (r + eps) * dec_r * dec_z
    ddrdy = -0.3 * y / (r + eps) * dec_r * dec_z
    ddrdz = dec_r * (-0.2 * z) * dec_z
    dfdx = cp * decay * dphidx + sp * ddrdx
    dfdy = cp * decay * dphidy + sp * ddrdy
    dfdz = cp * decay * dphidz + sp * ddrdz
    return v, (dfdx, dfdy, dfdz)


@register(
    "trig_product", "cos(x)*sin(y+z) + sin(xy)*cos(z)  —  non-separable cross-terms"
)
def f_trig_product(x, y, z):
    v = np.cos(x) * np.sin(y + z) + np.sin(x * y) * np.cos(z)
    dfdx = -np.sin(x) * np.sin(y + z) + y * np.cos(x * y) * np.cos(z)
    dfdy = np.cos(x) * np.cos(y + z) + x * np.cos(x * y) * np.cos(z)
    dfdz = np.cos(x) * np.cos(y + z) - np.sin(x * y) * np.sin(z)
    return v, (dfdx, dfdy, dfdz)


@register("radial_shell", "Thin Gaussian shell at r=π/2: exp(-20*(r - π/2)²)")
def f_radial_shell(x, y, z):
    eps = 1e-12
    r = np.sqrt(x**2 + y**2 + z**2)
    dr = r - np.pi / 2
    v = np.exp(-20 * dr**2)
    dvdr = v * (-40 * dr)
    drdx, drdy, drdz = x / (r + eps), y / (r + eps), z / (r + eps)
    return v, (dvdr * drdx, dvdr * drdy, dvdr * drdz)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualise AMS 3-D point interpolation with optional gradient data.\n"
            "Results are shown as orthogonal cross-sections through the domain centre."
        )
    )
    parser.add_argument(
        "--function",
        default="sin_cos_sin",
        choices=list(FUNCTIONS.keys()),
        metavar="NAME",
        help="Ground-truth function (default: sin_cos_sin). Use --list-functions to see all.",
    )
    parser.add_argument(
        "--list-functions",
        action="store_true",
        help="Print available functions and exit.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50000,
        help="Number of scattered 3-D samples (default: 50000)",
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
        "--grid_size",
        type=int,
        default=64,
        help="Evaluation grid resolution per axis (default: 64)",
    )
    parser.add_argument(
        "--slice_axis",
        default="all",
        choices=["x", "y", "z", "all"],
        help="Axis to slice through for visualisation (default: all)",
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_grid(domain, n):
    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = domain
    x1 = np.linspace(x_lo, x_hi, n)
    y1 = np.linspace(y_lo, y_hi, n)
    z1 = np.linspace(z_lo, z_hi, n)
    X, Y, Z = np.meshgrid(x1, y1, z1, indexing="ij")
    return x1, y1, z1, X, Y, Z


def centre_slice(arr3d, axis):
    i = arr3d.shape[axis] // 2
    return arr3d.take(i, axis=axis)


def add_slice_row(
    fig,
    axs_row,
    true3d,
    interp3d,
    axis,
    hlbl,
    vlbl,
    domain,
):
    sl_true = centre_slice(true3d, axis)
    sl_interp = centre_slice(interp3d, axis)
    abs_err = np.abs(sl_interp - sl_true)

    d = domain
    ext_map = {
        0: [d[2], d[3], d[4], d[5]],
        1: [d[0], d[1], d[4], d[5]],
        2: [d[0], d[1], d[2], d[3]],
    }
    extent = ext_map[axis]
    vmin, vmax = sl_true.min(), sl_true.max()

    for ax, data, title, cmap, v0, v1 in zip(
        axs_row,
        [sl_true, sl_interp, abs_err],
        ["Ground Truth", "Interpolated", "Absolute Error"],
        ["viridis", "viridis", "hot"],
        [vmin, vmin, None],
        [vmax, vmax, None],
    ):
        kw = dict(extent=extent, origin="lower", aspect="auto", cmap=cmap)
        if v0 is not None:
            kw["vmin"] = v0
            kw["vmax"] = v1
        im = ax.imshow(data.T, **kw)
        ax.set_xlabel(hlbl)
        ax.set_ylabel(vlbl)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    if args.list_functions:
        print("\nAvailable 3-D ground-truth functions:\n")
        max_len = max(len(k) for k in FUNCTIONS)
        for name, meta in FUNCTIONS.items():
            d = meta["domain"]
            print(f"  {name:<{max_len}}   {meta['description']}")
            print(
                f"  {'':<{max_len}}   domain: x∈[{d[0]:.2g},{d[1]:.2g}]  "
                f"y∈[{d[2]:.2g},{d[3]:.2g}]  z∈[{d[4]:.2g},{d[5]:.2g}]\n"
            )
        return 0

    entry = FUNCTIONS[args.function]
    f = entry["fn"]
    grad_f = entry["grad"]
    domain = entry["domain"]
    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = domain
    n = args.grid_size

    grad_tag = (
        f"  +gradients ({int(args.grad_fraction * 100)} % of points)"
        if args.use_gradients
        else "  (no gradients)"
    )
    print(f"\n[function]  {args.function}: {entry['description']}")
    print(f"[domain]    x∈[{x_lo},{x_hi}]  y∈[{y_lo},{y_hi}]  z∈[{z_lo},{z_hi}]")
    print(f"[samples]   {args.num_samples}{grad_tag}")
    print(f"[grid]      {n}³ = {n**3:,} eval points")

    # Ground-truth grid
    x1, y1, z1, X, Y, Z = make_grid(domain, n)
    Z_true_3d = f(X, Y, Z)

    # Scattered samples
    rng = np.random.default_rng()
    xs = rng.uniform(x_lo, x_hi, args.num_samples)
    ys = rng.uniform(y_lo, y_hi, args.num_samples)
    zs_coord = rng.uniform(z_lo, z_hi, args.num_samples)
    vals = f(xs, ys, zs_coord)
    points = np.column_stack((xs, ys, zs_coord))

    # Gradients
    if args.use_gradients:
        n_grad = max(1, int(args.num_samples * args.grad_fraction))
        grad_idx = rng.choice(args.num_samples, size=n_grad, replace=False)
        grad_pts = points[grad_idx]  # (n_grad, 3)
        gx, gy, gz = grad_f(xs[grad_idx], ys[grad_idx], zs_coord[grad_idx])
        gradients = np.column_stack((gx, gy, gz))  # (n_grad, 3)
        print(f"[gradients] providing {n_grad} gradient vectors")
    else:
        grad_pts = np.array([])
        gradients = np.array([])

    # Build the interpolant
    interp = ams_point_interpolant_3d(
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
    eval_pts = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
    Z_interp_3d = interp.eval(eval_pts).reshape(n, n, n)

    # Full-volume error metrics
    abs_error = np.abs(Z_interp_3d - Z_true_3d)
    mae = np.mean(abs_error)
    rmse = np.sqrt(np.mean((Z_interp_3d - Z_true_3d) ** 2))
    max_err = np.max(abs_error)
    print(f"\nMAE      : {mae:.6f}")
    print(f"RMSE     : {rmse:.6f}")
    print(f"Max error: {max_err:.6f}")

    # Determine slices
    slice_defs = {
        "z": [(2, "x", "y", xs, ys)],
        "y": [(1, "x", "z", xs, zs_coord)],
        "x": [(0, "y", "z", ys, zs_coord)],
    }
    all_slices = [
        (2, "x", "y", xs, ys),
        (1, "x", "z", xs, zs_coord),
        (0, "y", "z", ys, zs_coord),
    ]
    slices = all_slices if args.slice_axis == "all" else slice_defs[args.slice_axis]

    n_rows = len(slices)
    fig, axs = plt.subplots(n_rows, 3, figsize=(15, 4.5 * n_rows))
    if n_rows == 1:
        axs = axs[np.newaxis, :]

    axis_plane = {0: "YZ", 1: "XZ", 2: "XY"}
    fig.suptitle(
        f"{args.function}{grad_tag}  —  {entry['description']}\n"
        f"samples={args.num_samples}   grid={n}³   "
        f"MAE={mae:.3e}   RMSE={rmse:.3e}   max={max_err:.3e}",
        fontsize=10,
    )

    for row, (axis, hlbl, vlbl, sh, sv) in enumerate(slices):
        mid_val = [x1, y1, z1][axis][n // 2]
        axs[row, 0].set_ylabel(
            f"{axis_plane[axis]} slice\n@ {'xyz'[axis]}={mid_val:.2f}\n{vlbl}",
            fontsize=8,
        )
        add_slice_row(
            fig,
            axs[row],
            Z_true_3d,
            Z_interp_3d,
            axis=axis,
            hlbl=hlbl,
            vlbl=vlbl,
            domain=domain,
        )

    plt.tight_layout()
    plt.show(block=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
