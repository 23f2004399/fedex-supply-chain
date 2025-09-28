from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]       
CSV_DIR = PROJECT_ROOT / "csv_results"
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED = {
    "seed0": CSV_DIR / "dqn_v0_seed0.csv",
    "seed1": CSV_DIR / "dqn_v0_seed1.csv",
    "seed2": CSV_DIR / "dqn_v0_seed2.csv",
    "aggregate": CSV_DIR / "dqn_v0_aggregate.csv",
    "stability": CSV_DIR / "dqn_v0_stability_summary.csv",
}

# ---------- Helpers ----------
def _must_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"[ERROR] Missing file: {path}")
    try:
        return pd.read_csv(path)
    except Exception as e:
        sys.exit(f"[ERROR] Failed to read {path}: {e}")

def _save_show_close(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[saved] {path.relative_to(PROJECT_ROOT)}")

df_seed0 = _must_read(REQUIRED["seed0"])
df_seed1 = _must_read(REQUIRED["seed1"])
df_seed2 = _must_read(REQUIRED["seed2"])
df_agg   = _must_read(REQUIRED["aggregate"])
df_stab  = _must_read(REQUIRED["stability"])

plt.figure(figsize=(10, 6))
for name, df in [("seed0", df_seed0), ("seed1", df_seed1), ("seed2", df_seed2)]:
    if {"episode", "total_reward"}.issubset(df.columns):
        plt.plot(df["episode"], df["total_reward"], label=name, linewidth=1.4)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN: Per-seed raw episode total reward")
plt.grid(True, alpha=0.3)
plt.legend()
_save_show_close(FIG_DIR / "dqn_seeds_rewards_raw.png")

plt.figure(figsize=(10, 6))
for name, df in [("seed0", df_seed0), ("seed1", df_seed1), ("seed2", df_seed2)]:
    if {"episode", "return_ma50"}.issubset(df.columns):
        plt.plot(df["episode"], df["return_ma50"], label=f"{name} (MA50)", linewidth=1.6)
plt.xlabel("Episode")
plt.ylabel("Return (MA50)")
plt.title("DQN: Per-seed smoothed return (moving average 50)")
plt.grid(True, alpha=0.3)
plt.legend()
_save_show_close(FIG_DIR / "dqn_seeds_rewards_ma50.png")

if {"episode", "reward_mean", "reward_std"}.issubset(df_stab.columns):
    x = df_stab["episode"]
    m = df_stab["reward_mean"]
    s = df_stab["reward_std"]
    plt.figure(figsize=(10, 6))
    plt.plot(x, m, label="Mean total reward", linewidth=1.8)
    plt.fill_between(x, m - s, m + s, alpha=0.2, label="±1 std")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN: Aggregate performance (mean ± std across seeds)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    _save_show_close(FIG_DIR / "dqn_aggregate_mean_std_from_stability.png")

if {"episode", "epsilon"}.issubset(df_agg.columns):
    eps = df_agg.groupby("episode", as_index=True)["epsilon"].mean()
    plt.figure(figsize=(10, 6))
    plt.plot(eps.index, eps.values, linewidth=1.8)
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("DQN: Epsilon schedule (mean across seeds)")
    plt.grid(True, alpha=0.3)
    _save_show_close(FIG_DIR / "dqn_epsilon_schedule.png")

plt.figure(figsize=(10, 6))
any_loss = False
for name, df in [("seed0", df_seed0), ("seed1", df_seed1), ("seed2", df_seed2)]:
    cols = {"episode", "mean_loss"}
    if cols.issubset(df.columns) and df["mean_loss"].notna().any():
        any_loss = True
        plt.plot(df["episode"], df["mean_loss"], label=name, linewidth=1.2)
if any_loss:
    plt.xlabel("Episode")
    plt.ylabel("Mean Loss")
    plt.title("DQN: Mean loss per episode (by seed)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    _save_show_close(FIG_DIR / "dqn_losses_by_seed.png")
else:
    plt.close()
    print("[info] Skipped loss plot: mean_loss was empty/NaN in provided files.")

plt.figure(figsize=(10, 6))
for name, df in [("seed0", df_seed0), ("seed1", df_seed1), ("seed2", df_seed2)]:
    if {"episode", "wall_clock_s"}.issubset(df.columns):
        plt.plot(df["episode"], df["wall_clock_s"], label=name, linewidth=1.2)
plt.xlabel("Episode")
plt.ylabel("Wall clock (s)")
plt.title("DQN: Wall clock time per episode (by seed)")
plt.grid(True, alpha=0.3)
plt.legend()
_save_show_close(FIG_DIR / "dqn_wallclock_by_seed.png")

print("\nDone. Figures saved in ./figures")
