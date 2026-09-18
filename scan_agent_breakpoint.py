import argparse
import csv
import os
import time

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in os.sys.path:
    os.sys.path.insert(0, SCRIPT_DIR)
VEHICLE_SIM_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "CDC-2025-vehicle_simulation")
NUMERICAL_SIM_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "CDC-2025-numerical_example")
for extra_dir in (VEHICLE_SIM_DIR, NUMERICAL_SIM_DIR):
    if os.path.isdir(extra_dir) and extra_dir not in os.sys.path:
        os.sys.path.insert(0, extra_dir)


def estimate_break_status(
    num_agents,
    full_communication=True,
    grid_budget=1e4,
    max_pred_cov_mem_gb=2.0,
):
    if full_communication:
        n_dimensions = num_agents
    else:
        # With nearest-neighbor communication in current implementation,
        # each local GP is at most 3D for interior agents.
        n_dimensions = 3 if num_agents >= 3 else num_agents

    points_per_axis = int(grid_budget ** (1 / n_dimensions)) if n_dimensions > 0 else 0
    grid_points = points_per_axis ** n_dimensions if points_per_axis > 0 else 0

    # Rough lower bound of one dense predictive covariance matrix in float32.
    pred_cov_mem_gb = (grid_points * grid_points * 4) / (1024 ** 3)

    if points_per_axis <= 1:
        status = "BROKEN_DEGENERATE_GRID"
        reason = "points_per_axis <= 1, BO discretization collapses to one point."
    elif pred_cov_mem_gb > max_pred_cov_mem_gb:
        status = "BROKEN_OOM_RISK"
        reason = f"Predictive covariance estimate {pred_cov_mem_gb:.2f}GB exceeds budget."
    elif full_communication and num_agents >= 11:
        status = "HIGH_RISK"
        reason = "Full communication + high dimension usually yields poor exploration."
    else:
        status = "OK"
        reason = "No immediate structural break detected."

    return {
        "num_agents": num_agents,
        "n_dimensions": n_dimensions,
        "points_per_axis": points_per_axis,
        "grid_points": grid_points,
        "pred_cov_mem_gb": pred_cov_mem_gb,
        "status": status,
        "reason": reason,
    }


def choose_next_x(cube, x_sample, torch_module):
    candidate_mask = torch_module.logical_or(cube.M, cube.G)
    if torch_module.any(candidate_mask):
        ucb_candidates = cube.ucb[candidate_mask]
        max_indices = torch_module.nonzero(
            ucb_candidates == torch_module.max(ucb_candidates), as_tuple=True
        )[0]
        random_max_index = max_indices[torch_module.randint(len(max_indices), (1,))].item()
        return cube.discr_domain[candidate_mask][random_max_index, :]
    return torch_module.cat((x_sample[-1], cube.iteration.flatten() + 1))


def run_runtime_probe(
    num_agents,
    full_communication,
    iterations,
    seed,
    grid_budget,
):
    try:
        import torch
        from pacsbo_main import PACSBO, GPRegressionModel, compute_X_plot
    except ModuleNotFoundError as exc:
        return {
            "num_agents": num_agents,
            "status": "BROKEN_MISSING_DEPENDENCY",
            "reason": f"Missing dependency: {exc}",
            "elapsed_sec": 0.0,
            "final_samples": -1,
            "n_dimensions": num_agents if full_communication else (3 if num_agents >= 3 else num_agents),
            "points_per_axis": 0,
            "grid_points": 0,
        }

    np.random.seed(seed + num_agents)
    torch.manual_seed(seed + num_agents)

    if full_communication:
        n_dimensions = num_agents
    else:
        n_dimensions = 3 if num_agents >= 3 else num_agents

    points_per_axis = int(grid_budget ** (1 / n_dimensions)) if n_dimensions > 0 else 0
    if points_per_axis <= 1:
        return {
            "num_agents": num_agents,
            "status": "BROKEN_DEGENERATE_GRID",
            "reason": "points_per_axis <= 1, runtime probe skipped.",
            "elapsed_sec": 0.0,
            "final_samples": 0,
            "n_dimensions": n_dimensions,
            "points_per_axis": points_per_axis,
            "grid_points": 1 if points_per_axis == 1 else 0,
        }

    grid_points = points_per_axis ** n_dimensions
    start = time.perf_counter()

    try:
        X_plot = compute_X_plot(n_dimensions=n_dimensions, points_per_axis=points_per_axis)

        noise_std = 1e-2
        delta_confidence = 0.9
        exploration_threshold = 0.1
        RKHS_norm = 1.0
        lengthscale_agent_spatio = 0.3
        a_parameter = 50.0
        lengthscale_temporal_RBF = 20.0
        lengthscale_temporal_Ma12 = 5.0
        output_variance_RBF = 0.1
        output_variance_Ma12 = 0.1
        safety_threshold = -np.inf

        # Each sample is [a1, ..., an, time].
        x0 = torch.rand(n_dimensions).unsqueeze(0)
        y0 = torch.tensor([0.0], dtype=torch.float32)
        X_sample = x0.clone()
        Y_sample = y0.clone()

        for t in range(1, iterations + 1):
            iteration_for_kernel = t
            cube = PACSBO(
                delta_confidence=delta_confidence,
                noise_std=noise_std,
                tuple_ik=(-1, -1),
                X_plot=X_plot,
                X_sample=X_sample,
                Y_sample=Y_sample,
                iteration=iteration_for_kernel,
                safety_threshold=safety_threshold,
                exploration_threshold=exploration_threshold,
                compute_all_sets=False,
                lengthscale_agent_spatio=lengthscale_agent_spatio,
                a_parameter=a_parameter,
                lengthscale_temporal_RBF=lengthscale_temporal_RBF,
                lengthscale_temporal_Ma12=lengthscale_temporal_Ma12,
                output_variance_RBF=output_variance_RBF,
                output_variance_Ma12=output_variance_Ma12,
            )
            cube.compute_model(gpr=GPRegressionModel)
            cube.compute_mean_var()
            cube.compute_confidence_intervals_evaluation(RKHS_norm_guessed=RKHS_norm)
            cube.compute_safe_set()
            cube.maximizer_routine(best_lower_bound_others=-np.inf)
            cube.expander_routine()

            x_new = choose_next_x(cube, X_sample, torch)
            x_new_spatial = x_new[:-1].unsqueeze(0)

            # Synthetic reward probe to keep runtime test lightweight.
            y_new = torch.tensor([float(torch.sum(x_new_spatial) / n_dimensions)], dtype=torch.float32)
            X_sample = torch.cat((X_sample, x_new_spatial), dim=0)
            Y_sample = torch.cat((Y_sample, y_new), dim=0)

        elapsed = time.perf_counter() - start
        return {
            "num_agents": num_agents,
            "status": "OK",
            "reason": "Runtime probe finished.",
            "elapsed_sec": elapsed,
            "final_samples": int(Y_sample.shape[0]),
            "n_dimensions": n_dimensions,
            "points_per_axis": points_per_axis,
            "grid_points": grid_points,
        }
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return {
            "num_agents": num_agents,
            "status": "BROKEN_RUNTIME_EXCEPTION",
            "reason": f"{type(exc).__name__}: {exc}",
            "elapsed_sec": elapsed,
            "final_samples": -1,
            "n_dimensions": n_dimensions,
            "points_per_axis": points_per_axis,
            "grid_points": grid_points,
        }


def save_sweep_plot(rows, mode, output_path):
    x_vals = [int(r["num_agents"]) for r in rows]
    status_vals = [r["status"] for r in rows]

    if mode == "estimate":
        y_vals = [float(r["pred_cov_mem_gb"]) for r in rows]
        y_label = "Estimated predictive covariance memory (GB)"
        title = "Agent Sweep (Estimate Mode)"
    else:
        y_vals = [float(r["elapsed_sec"]) for r in rows]
        y_label = "Runtime probe elapsed time (sec)"
        title = "Agent Sweep (Runtime Mode)"

    color_map = {
        "OK": "tab:green",
        "HIGH_RISK": "tab:orange",
        "BROKEN_DEGENERATE_GRID": "tab:red",
        "BROKEN_OOM_RISK": "tab:red",
        "BROKEN_RUNTIME_EXCEPTION": "tab:red",
        "BROKEN_MISSING_DEPENDENCY": "tab:red",
    }
    colors = [color_map.get(s, "tab:gray") for s in status_vals]

    plt.figure(figsize=(10, 5))
    plt.scatter(x_vals, y_vals, c=colors, s=60)
    plt.plot(x_vals, y_vals, alpha=0.35, color="tab:blue")

    # Annotate first broken point for quick read.
    for idx, status in enumerate(status_vals):
        if status.startswith("BROKEN"):
            plt.axvline(x=x_vals[idx], linestyle="--", color="tab:red", alpha=0.6)
            plt.text(
                x_vals[idx],
                max(y_vals) * 0.85 if len(y_vals) > 0 else 0.0,
                f"first broken: {x_vals[idx]}",
                color="tab:red",
                rotation=90,
                va="center",
            )
            break

    plt.xlabel("Number of agents")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)

    legend_entries = []
    seen = set()
    for s in status_vals:
        if s not in seen:
            seen.add(s)
            legend_entries.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color_map.get(s, "tab:gray"),
                    markersize=8,
                    label=s,
                )
            )
    if legend_entries:
        plt.legend(handles=legend_entries, loc="best")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Scan likely breakpoints for MAS SafeOpt setup."
    )
    parser.add_argument("--min-agents", type=int, default=2)
    parser.add_argument("--max-agents", type=int, default=40)
    parser.add_argument(
        "--full-communication",
        action="store_true",
        default=False,
        help="Use full communication mode (dimension = num_agents).",
    )
    parser.add_argument(
        "--grid-budget",
        type=float,
        default=1e4,
        help="Budget used in points_per_axis=int(grid_budget**(1/n)).",
    )
    parser.add_argument(
        "--max-pred-cov-mem-gb",
        type=float,
        default=2.0,
        help="Memory threshold for predictive covariance matrix estimate.",
    )
    parser.add_argument(
        "--mode",
        choices=["estimate", "runtime"],
        default="estimate",
        help="estimate: structural check; runtime: short real GP run probe.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Iterations for runtime probe mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=107,
        help="Random seed for runtime probe mode.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="Optional CSV output path.",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default="",
        help="Optional PNG output path for sweep plot.",
    )
    args = parser.parse_args()

    if args.mode == "estimate":
        header = [
            "num_agents",
            "status",
            "n_dimensions",
            "points_per_axis",
            "grid_points",
            "pred_cov_mem_gb",
            "reason",
        ]
    else:
        header = [
            "num_agents",
            "status",
            "n_dimensions",
            "points_per_axis",
            "grid_points",
            "elapsed_sec",
            "final_samples",
            "reason",
        ]

    print(",".join(header))
    first_broken = None
    rows = []
    for n in range(args.min_agents, args.max_agents + 1):
        if args.mode == "estimate":
            row = estimate_break_status(
                num_agents=n,
                full_communication=args.full_communication,
                grid_budget=args.grid_budget,
                max_pred_cov_mem_gb=args.max_pred_cov_mem_gb,
            )
            row_out = {
                "num_agents": row["num_agents"],
                "status": row["status"],
                "n_dimensions": row["n_dimensions"],
                "points_per_axis": row["points_per_axis"],
                "grid_points": row["grid_points"],
                "pred_cov_mem_gb": f"{row['pred_cov_mem_gb']:.4f}",
                "reason": row["reason"],
            }
        else:
            row = run_runtime_probe(
                num_agents=n,
                full_communication=args.full_communication,
                iterations=args.iterations,
                seed=args.seed,
                grid_budget=args.grid_budget,
            )
            row_out = {
                "num_agents": n,
                "status": row["status"],
                "n_dimensions": row["n_dimensions"],
                "points_per_axis": row["points_per_axis"],
                "grid_points": row["grid_points"],
                "elapsed_sec": f"{row['elapsed_sec']:.4f}",
                "final_samples": row["final_samples"],
                "reason": row["reason"],
            }

        rows.append(row_out)
        print(",".join(str(row_out[h]) for h in header))

        if first_broken is None and row_out["status"].startswith("BROKEN"):
            first_broken = row_out["num_agents"]

    if first_broken is None:
        print("first_broken_agent_count=none_in_range")
    else:
        print(f"first_broken_agent_count={first_broken}")

    if args.output_csv:
        csv_path = args.output_csv
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(SCRIPT_DIR, csv_path)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        print(f"csv_saved={csv_path}")

    if args.output_plot:
        plot_path = args.output_plot
        if not os.path.isabs(plot_path):
            plot_path = os.path.join(SCRIPT_DIR, plot_path)
        save_sweep_plot(rows=rows, mode=args.mode, output_path=plot_path)
        print(f"plot_saved={plot_path}")


if __name__ == "__main__":
    main()
