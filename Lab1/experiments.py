"""Run reproducible Q-learning and SARSA experiments on FrozenLake.

Default run:
    python experiments.py

The default configuration performs 100,000 episodes for each agent/seed
combination and writes plots, a summary, and a four-page PDF report to outputs/.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from qlearning import make_env, train


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
EPISODES_DEFAULT = 100_000
SEEDS = (0, 1, 2, 3, 4)
ALPHAS = (0.01, 0.05, 0.1, 0.3, 0.7)
ALGORITHMS = ("q_learning", "sarsa")
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
DECAY_FRACTION = 0.8
MOVING_AVERAGE_WINDOW = 500


def _run_one_job(job):
    """Worker function kept at module level for Windows multiprocessing."""
    algorithm, alpha, seed, episodes, keep_returns = job
    env = make_env()

    try:
        _, returns, _, _ = train(
            env=env,
            episodes=episodes,
            alpha=alpha,
            gamma=GAMMA,
            eps_start=EPS_START,
            eps_end=EPS_END,
            algorithm=algorithm,
            seed=seed,
            snapshot_interval=0,
            decay_fraction=DECAY_FRACTION,
        )
    finally:
        env.close()

    returns = np.asarray(returns, dtype=np.float32)
    result = {
        "algorithm": algorithm,
        "alpha": alpha,
        "seed": seed,
        "success_rate": float(np.mean(returns[-1000:])),
    }
    if keep_returns:
        result["returns"] = returns
    return result


def _run_jobs(jobs, workers, title):
    """Execute experiment jobs and print simple progress in the terminal."""
    results = []
    if workers == 1:
        for index, job in enumerate(jobs, start=1):
            results.append(_run_one_job(job))
            print(f"{title}: {index}/{len(jobs)} complete")
        return results

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run_one_job, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            print(f"{title}: {index}/{len(jobs)} complete")

    return results


def _moving_average(values, window):
    window = max(1, min(window, len(values)))
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="valid")


def _learning_curve_figure(curves, episodes):
    window = min(MOVING_AVERAGE_WINDOW, episodes)
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.94, "1. TRAINING PERFORMANCE",
             fontsize=17, weight="bold")
    fig.text(
        0.08, 0.89,
        f"Each method was trained for {episodes:,} episodes using five "
        f"independent random seeds ({', '.join(map(str, SEEDS))}). "
        "The learning rate for this comparison was 0.1.",
        fontsize=10.5, wrap=True, va="top",
    )
    ax = fig.add_axes([0.10, 0.39, 0.82, 0.40])

    for algorithm, label in (
        ("q_learning", "Q-learning"),
        ("sarsa", "SARSA"),
    ):
        smoothed = np.stack([
            _moving_average(curves[(algorithm, seed)], window)
            for seed in SEEDS
        ])
        mean = smoothed.mean(axis=0)
        std = smoothed.std(axis=0)
        x = np.arange(window, episodes + 1)

        ax.plot(x, mean, label=label, linewidth=1.8)
        ax.fill_between(x, mean - std, mean + std, alpha=0.18)

    ax.set(
        title="Mean learning curve with variation across seeds",
        xlabel="Episode",
        ylabel=f"Success rate ({window}-episode moving average)",
        ylim=(0, 1),
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.text(
        0.10, 0.29,
        "The plotted line is the mean moving success rate across the five "
        "seeds. The shaded region represents one standard deviation. "
        "Differences between the curves show how the learning strategy "
        "and exploration behaviour affect progress.",
        fontsize=10.5, wrap=True, va="top", linespacing=1.5,
    )
    return fig


def _alpha_sweep_figure(alpha_summary):
    x = np.arange(len(ALPHAS))
    width = 0.36
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.94, "2. LEARNING-RATE EXPERIMENT",
             fontsize=17, weight="bold")
    fig.text(
        0.08, 0.89,
        "Five learning rates (0.01, 0.05, 0.1, 0.3, and 0.7) were evaluated "
        "with the same five random seeds. Each bar shows the mean success "
        "rate over the final 1,000 episodes; error bars show standard deviation.",
        fontsize=10.5, wrap=True, va="top",
    )
    ax = fig.add_axes([0.11, 0.42, 0.80, 0.36])

    for offset, algorithm, label in (
        (-width / 2, "q_learning", "Q-learning"),
        (width / 2, "sarsa", "SARSA"),
    ):
        means = [alpha_summary[(algorithm, alpha)][0] for alpha in ALPHAS]
        stds = [alpha_summary[(algorithm, alpha)][1] for alpha in ALPHAS]
        ax.bar(
            x + offset, means, width, yerr=stds, capsize=4,
            label=label, alpha=0.9,
        )

    ax.set(
        title="Final success rate by learning rate",
        xlabel="Learning rate (alpha)",
        ylabel="Success rate over final 1,000 episodes",
        xticks=x,
        xticklabels=[str(alpha) for alpha in ALPHAS],
        ylim=(0, 1),
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.text(
        0.10, 0.30,
        "The learning rate determines how much each new temporal-difference "
        "target changes the existing estimate. Very small values can learn "
        "slowly; very large values can make learning unstable. The results "
        "show how these effects differ between the two algorithms.",
        fontsize=10.5, wrap=True, va="top", linespacing=1.5,
    )
    return fig


def _write_report(curves, alpha_summary, seed0_rate, episodes):
    """Create a four-page report from the results generated in this run."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    learning_fig = _learning_curve_figure(curves, episodes)
    alpha_fig = _alpha_sweep_figure(alpha_summary)

    learning_fig.savefig(OUTPUT_DIR / "learning_curves.png", dpi=180)
    alpha_fig.savefig(OUTPUT_DIR / "alpha_sweep.png", dpi=180)

    q_mean, q_std = alpha_summary[("q_learning", 0.1)]
    s_mean, s_std = alpha_summary[("sarsa", 0.1)]
    q_best = max(ALPHAS, key=lambda a: alpha_summary[("q_learning", a)][0])
    s_best = max(ALPHAS, key=lambda a: alpha_summary[("sarsa", a)][0])

    with PdfPages(OUTPUT_DIR / "report.pdf") as pdf:
        # Page 1: problem and environment.
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.94, "AICL 407 | REINFORCEMENT LEARNING",
                 fontsize=12, weight="bold")
        fig.text(0.08, 0.89, "FrozenLake: Q-learning and SARSA",
                 fontsize=23, weight="bold")
        fig.text(0.08, 0.85, "Lab 1 — Tabular Temporal-Difference Learning",
                 fontsize=13)
        fig.text(0.08, 0.80, "Student: ______________________________",
                 fontsize=10.5)

        sections = [
            (0.73, "Objective",
             "Implement and compare two tabular temporal-difference methods "
             "on the slippery 8×8 FrozenLake environment."),
            (0.61, "Environment",
             "The agent observes one of 64 states and selects one of four "
             "actions. Movement is stochastic because the lake is slippery. "
             "Reaching the goal gives +1 reward; other transitions give 0. "
             "A hole or the goal ends an episode. A 200-step time limit is "
             "used to allow longer trajectories."),
            (0.43, "Learning task",
             "The agent must learn action values from trial-and-error "
             "interaction. Performance is evaluated using the proportion of "
             "episodes in which the goal is reached."),
            (0.30, "Update rules",
             "Q-learning is off-policy and uses the maximum next-state value:\n"
             "Q(s,a) ← Q(s,a) + α[r + γ maxₐ′ Q(s′,a′) − Q(s,a)]\n\n"
             "SARSA is on-policy and uses the next action selected by the policy:\n"
             "Q(s,a) ← Q(s,a) + α[r + γ Q(s′,a′) − Q(s,a)]\n\n"
             "For true terminal transitions, the target is the immediate reward."),
        ]
        for y, heading, body in sections:
            fig.text(0.08, y, heading, fontsize=13, weight="bold", va="top")
            fig.text(0.08, y - 0.035, body, fontsize=10.5, va="top",
                     wrap=True, linespacing=1.5)

        fig.text(0.08, 0.06,
                 "The implementation and the numerical results are generated independently in this repository.",
                 fontsize=8, color="dimgray")
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: learning curves.
        pdf.savefig(learning_fig)
        plt.close(learning_fig)

        # Page 3: learning-rate comparison.
        pdf.savefig(alpha_fig)
        plt.close(alpha_fig)

        # Page 4: findings and conclusion.
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.94, "3. RESULTS AND CONCLUSION", fontsize=17, weight="bold")
        findings = [
            ("Task 4 check",
             f"Q-learning with seed 0 achieved a success rate of {seed0_rate:.3f} "
             "over the final 1,000 episodes. The target specified for this "
             f"experiment is 0.700 or higher: "
             f"{'PASS' if seed0_rate >= 0.7 else 'NOT YET MET'}."),
            ("Learning-rate comparison",
             f"The strongest mean result for Q-learning occurred at alpha={q_best}; "
             f"for SARSA it occurred at alpha={s_best}. These values are selected "
             "from the five rates tested in this experiment."),
            ("Interpretation",
             "A small learning rate makes updates gradual and can require more "
             "experience. A large learning rate reacts quickly to new samples "
             "but can make estimates unstable. The best setting depends on the "
             "agent, stochastic transitions, and exploration schedule."),
            ("Conclusion",
             "Both methods learn from sampled transitions without requiring a "
             "model of the environment. Q-learning evaluates the best estimated "
             "next action, whereas SARSA evaluates the next action chosen by its "
             "policy. The learning curves and learning-rate sweep provide two "
             "complementary views of their behaviour."),
            ("Reproducibility",
             f"Environment: FrozenLake-v1, 8×8, slippery, 200-step limit. "
             f"Episodes per run: {episodes:,}. Seeds: {list(SEEDS)}. "
             f"Gamma: {GAMMA}; epsilon: {EPS_START} to {EPS_END}, linearly "
             f"decayed over the first {DECAY_FRACTION:.0%} of training. "
             "The plots and this report are generated from the experiment "
             "results saved during the current run."),
        ]
        y = 0.86
        for heading, body in findings:
            fig.text(0.08, y, heading, fontsize=13, weight="bold", va="top")
            fig.text(0.08, y - 0.04, body, fontsize=10.5, va="top",
                     wrap=True, linespacing=1.5)
            y -= 0.16
        fig.text(0.08, 0.06, "Generated with Python, NumPy, Gymnasium, and Matplotlib.",
                 fontsize=8, color="dimgray")
        pdf.savefig(fig)
        plt.close(fig)

    plt.close(learning_fig)
    plt.close(alpha_fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes", type=int, default=EPISODES_DEFAULT,
        help=f"episodes per run (default: {EPISODES_DEFAULT})",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="number of parallel worker processes (default: 4; use 1 for sequential)",
    )
    args = parser.parse_args()

    if args.episodes < 1000:
        parser.error("--episodes must be at least 1000 to report the final-1000 success rate")
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # The alpha=0.1 runs are also used for the learning-curve comparison.
    baseline_jobs = [
        (algorithm, 0.1, seed, args.episodes, True)
        for algorithm in ALGORITHMS
        for seed in SEEDS
    ]
    print("Running learning-curve experiments...")
    baseline_results = _run_jobs(baseline_jobs, args.workers, "Learning curves")

    curves = {
        (result["algorithm"], result["seed"]): result["returns"]
        for result in baseline_results
    }
    seed0_rate = next(
        result["success_rate"] for result in baseline_results
        if result["algorithm"] == "q_learning" and result["seed"] == 0
    )

    # The alpha=0.1 results already exist, so only the other four rates are run.
    sweep_jobs = [
        (algorithm, alpha, seed, args.episodes, False)
        for algorithm in ALGORITHMS
        for alpha in ALPHAS
        if alpha != 0.1
        for seed in SEEDS
    ]
    print("Running learning-rate sweep...")
    sweep_results = _run_jobs(sweep_jobs, args.workers, "Learning-rate sweep")

    all_sweep_results = list(baseline_results) + sweep_results
    alpha_summary = {}
    for algorithm in ALGORITHMS:
        for alpha in ALPHAS:
            values = [
                result["success_rate"] for result in all_sweep_results
                if result["algorithm"] == algorithm and result["alpha"] == alpha
            ]
            alpha_summary[(algorithm, alpha)] = (
                float(np.mean(values)), float(np.std(values))
            )

    summary_lines = [
        f"Environment: FrozenLake-v1 (8x8, slippery, 200-step limit)",
        f"Episodes per run: {args.episodes}",
        f"Seeds: {list(SEEDS)}",
        f"Gamma: {GAMMA}; epsilon: {EPS_START} -> {EPS_END}; "
        f"decay fraction: {DECAY_FRACTION}",
        "",
        f"Q-learning seed 0, final 1,000 episodes: {seed0_rate:.3f} "
        f"({'PASS' if seed0_rate >= 0.7 else 'NOT YET MET'}; target >= 0.700)",
        "",
        "Mean success rate over final 1,000 episodes (mean +/- std across seeds):",
    ]
    for algorithm in ALGORITHMS:
        for alpha in ALPHAS:
            mean, std = alpha_summary[(algorithm, alpha)]
            summary_lines.append(
                f"{algorithm:10s} alpha={alpha:<4}: {mean:.3f} +/- {std:.3f}"
            )

    (OUTPUT_DIR / "summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )
    _write_report(curves, alpha_summary, seed0_rate, args.episodes)

    print("\n".join(summary_lines))
    print(f"\nOutputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
