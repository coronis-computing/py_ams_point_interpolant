"""Console script for py_ams_point_interpolant"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from py_ams_point_interpolant.ams_point_interpolant import point_interpolant
from numpy import genfromtxt
import faulthandler

faulthandler.enable()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_samples",
        type=str,
        default="",
        help="Input points (ASCII file)",
    )
    parser.add_argument(
        "--delimiter", type=str, default=" ", help="Delimiter of input file"
    )
    args = parser.parse_args()

    return args


def main():
    """Console script for py_ams_point_interpolant"""
    args = parse_args()

    data = genfromtxt(args.input_samples, delimiter=args.delimiter)
    points = data[:, :2]
    values = data[:, 2]

    # Interpolate samples using AMS point interpolant
    point_interpolant(points, values)
    # Z_interp = ams_point_interpolant(points, values, (X, Y), method="cubic")
    # mask = ~np.isnan(Z_interp)  # griddata may return NaNs near borders

    # # Error metrics
    # abs_error = np.abs(Z_interp - Z_true)

    # mae = np.mean(abs_error[mask])
    # rmse = np.sqrt(np.mean((Z_interp[mask] - Z_true[mask]) ** 2))

    # print(f"MAE  : {mae:.6f}")
    # print(f"RMSE : {rmse:.6f}")

    # # Show results
    # fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    # im0 = axs[0].imshow(Z_true, extent=[-np.pi, np.pi, -np.pi, np.pi], origin="lower")
    # axs[0].set_title("Ground Truth")
    # plt.colorbar(im0, ax=axs[0])

    # im1 = axs[1].imshow(Z_interp, extent=[-np.pi, np.pi, -np.pi, np.pi], origin="lower")
    # axs[1].set_title("Interpolated")
    # plt.colorbar(im1, ax=axs[1])

    # im2 = axs[2].imshow(
    #     abs_error, extent=[-np.pi, np.pi, -np.pi, np.pi], origin="lower"
    # )
    # axs[2].set_title("Absolute Error")
    # plt.colorbar(im2, ax=axs[2])

    # plt.tight_layout()
    # plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
