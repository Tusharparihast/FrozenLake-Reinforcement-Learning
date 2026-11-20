"""
Task 5: DQN baseline and the no-target-network ablation, 5 seeds each.

Run with `python experiments.py`. Writes baseline_curve.png, no_target.png and
summary.txt to outputs/. The baseline run with seed 0 is the Task 4 check
(best 100-episode average return must be >= 400, with finite Q-values).
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dqn import train_dqn, moving_average

SEEDS = [0, 1, 2, 3, 4]
N_EPISODES = 500
SOLVED = 475
CONFIGS = {"baseline": 500, "no_target": 1}  # target sync interval in steps
LABELS = {"baseline": "Baseline (sync every 500 steps)", "no_target": "No target network (sync every step)"}
SMOOTH = 10      # episodes in the moving average of the returns
Q_SMOOTH = 500   # gradient steps in the moving average of the Q-values
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def run_one(job):
    config, seed = job
    return train_dqn(n_episodes=N_EPISODES, seed=seed, target_sync_every=CONFIGS[config])


def run_all_training():
    """Train every (config, seed) in parallel. Returns {config: {seed: result}}."""
    jobs = [(config, seed) for config in CONFIGS for seed in SEEDS]
    with ProcessPoolExecutor() as executor:
        outputs = list(executor.map(run_one, jobs))
    results = {config: {} for config in CONFIGS}
    for (config, seed), result in zip(jobs, outputs):
        results[config][seed] = result
    return results


def first_solved_episode(returns):
    """Episode at which the 100-episode average first reaches SOLVED, or None."""
    hits = np.flatnonzero(moving_average(returns, 100) >= SOLVED)
    return int(hits[0]) + 100 if len(hits) else None


def plot_returns(ax, results, config):
    smoothed = np.array([moving_average(results[config][s]["returns"], SMOOTH) for s in SEEDS])
    mean, std = smoothed.mean(axis=0), smoothed.std(axis=0)
    episodes = np.arange(SMOOTH, N_EPISODES + 1)
    ax.plot(episodes, mean, label=LABELS[config])
    ax.fill_between(episodes, mean - std, mean + std, alpha=0.25)


def plot_baseline(results):
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_returns(ax, results, "baseline")
    ax.axhline(SOLVED, color="gray", linestyle="--", label=f"Solved ({SOLVED})")
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Return (moving average, window {SMOOTH})")
    ax.set_title(f"DQN baseline on CartPole (mean and std over {len(SEEDS)} seeds)")
    ax.set_ylim(0, 520)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_no_target(results):
    """Left: returns of both configs. Right: mean max-Q per gradient step, one line per seed, log scale."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for config in CONFIGS:
        plot_returns(ax1, results, config)
    ax1.axhline(SOLVED, color="gray", linestyle="--")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel(f"Return (moving average, window {SMOOTH})")
    ax1.set_title("Return")
    ax1.set_ylim(0, 520)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper left", fontsize=8)

    for color, config in zip(["C0", "C1"], CONFIGS):
        for i, seed in enumerate(SEEDS):
            q = moving_average(results[config][seed]["mean_max_q"], Q_SMOOTH)
            ax2.plot(q, color=color, alpha=0.6, linewidth=0.8, label=LABELS[config] if i == 0 else None)
    ax2.set_yscale("symlog", linthresh=1)
    ax2.set_xlabel("Gradient step")
    ax2.set_ylabel(f"Mean max-Q (moving average, window {Q_SMOOTH}, log scale)")
    ax2.set_title("Q-value estimates (one line per seed)")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper left", fontsize=8)

    fig.suptitle(f"Effect of the target network ({len(SEEDS)} seeds)")
    fig.tight_layout()
    return fig


def write_summary(results, path):
    task4 = results["baseline"][0]
    best0 = moving_average(task4["returns"], 100).max()
    finite0 = bool(np.all(np.isfinite(task4["mean_max_q"])))
    lines = [
        f"Settings: {N_EPISODES} episodes, seeds {SEEDS}",
        f"Task 4 check (baseline, seed 0): best 100-episode average {best0:.1f}, needs 400; "
        f"Q-values finite {finite0} -> {'PASS' if best0 >= 400 and finite0 else 'FAIL'}",
        "",
    ]
    for config in CONFIGS:
        lines.append(LABELS[config])
        bests, firsts = [], []
        for seed in SEEDS:
            r = results[config][seed]
            best = moving_average(r["returns"], 100).max()
            first = first_solved_episode(r["returns"])
            bests.append(best)
            firsts.append(first)
            lines.append(
                f"  seed {seed}: best 100-ep average {best:6.1f}, "
                f"first reaches {SOLVED} at episode {first if first else 'never'}, "
                f"final eval return {r['eval_returns'][-1]:6.1f}, "
                f"max |mean max-Q| {np.abs(r['mean_max_q']).max():8.1f}, "
                f"Q finite {np.all(np.isfinite(r['mean_max_q']))}"
            )
        solved = [f for f in firsts if f is not None]
        lines.append(f"  best 100-ep average: mean {np.mean(bests):.1f}, std {np.std(bests):.1f}")
        lines.append(f"  first reaches {SOLVED}: " + (
            f"{len(solved)} of {len(SEEDS)} seeds, mean episode {np.mean(solved):.0f}" if solved else "no seed"))
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = run_all_training()
    plot_baseline(results).savefig(os.path.join(OUTPUT_DIR, "baseline_curve.png"), dpi=150)
    plot_no_target(results).savefig(os.path.join(OUTPUT_DIR, "no_target.png"), dpi=150)
    write_summary(results, os.path.join(OUTPUT_DIR, "summary.txt"))
    print(open(os.path.join(OUTPUT_DIR, "summary.txt")).read())


if __name__ == "__main__":
    main()