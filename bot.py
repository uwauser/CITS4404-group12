import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from optimizer.abc import artificial_bee_colony
from optimizer.pso import particle_swarm
from optimizer.gwo import grey_wolf
from optimizer.woa import whale
from optimizer.fa import firefly
from optimizer.cs import cuckoo_search
from optimizer.sa import simulated_annealing

from utils import quality, downsample
from config import settings

# --- Load & Prepare Data ---
df = pd.read_csv("BTCUSD.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

split_date = pd.to_datetime('2020-01-01')
train_data = df[df['date'] < split_date]
test_data = df[df['date'] >= split_date]

train_prices = train_data['close'].dropna().values
test_prices = test_data['close'].dropna().values

# --- Setup ---
optimizers = {
    "ABC": artificial_bee_colony,
    "PSO": particle_swarm,
    "GWO": grey_wolf,
    "WOA": whale,
    "FA": firefly,
    "CS": cuckoo_search,
    "SA": simulated_annealing
}

n_runs = 10
summary = []
all_results = {name: [] for name in optimizers}

# --- Run Experiments ---
for name, optimizer in optimizers.items():
    for idx, setting in enumerate(settings[name]):
        print(f"\nRunning {name} - Setting {idx+1}")
        results = []
        start_time = time.time()

        for run in range(n_runs):
            log = []
            best_params, train_profit, tf = optimizer(train_prices, log=log, setting=setting)
            test_ds_prices = downsample(test_prices, tf)
            info = quality(best_params[:-1], test_ds_prices, return_info=True)
            results.append({
                "train_profit": train_profit,
                "info": info,
                "params": best_params,
                "timeframe": tf,
                "log": log
            })

        exec_time = (time.time() - start_time) / n_runs
        train = [r["train_profit"] for r in results]
        test = [r["info"]["profit"] for r in results]
        buys = [len(r["info"]["buy_points"]) for r in results]
        sells = [len(r["info"]["sell_points"]) for r in results]
        convergence = [len(r["log"]) for r in results]
        timeframes = [r["timeframe"] for r in results]
        underperform = [p <= 1000 for p in test]
        drawdowns = [np.mean(r["info"]["drawdown"]) * 100 for r in results]

        summary.append({
            "Optimizer": name,
            "Setting ID": idx + 1,
            "Avg Train": round(np.mean(train), 2),
            "Avg": round(np.mean(test), 2),
            "Max": round(np.max(test), 2),
            "Min": round(np.min(test), 2),
            "Std": round(np.std(test), 2),
            "Avg Buys": round(np.mean(buys), 2),
            "Avg Sells": round(np.mean(sells), 2),
            "Avg Convergence": round(np.mean(convergence), 2),
            "Timeframe (hours)": max(set(timeframes), key=timeframes.count),
            "Underperform (%)": round(100 * np.mean(underperform), 1),
            "Avg Drawdown (%)": round(np.mean(drawdowns), 2),
            "Exec Time (s)": round(exec_time, 2)
        })
        all_results[name].append(results)

# --- Summary Table ---
summary_df = pd.DataFrame(summary)
print("\nSummary Table:")
print(summary_df.to_string(index=False))

n_settings = len(next(iter(all_results.values())))

# --- Convergence Subplots: All Settings in One Figure ---
n_cols = 2
n_rows = int(np.ceil(n_settings / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=False)
axes = axes.flatten()

for setting_idx in range(n_settings):
    ax = axes[setting_idx]
    for name in all_results:
        logs = [np.array(run["log"]) for run in all_results[name][setting_idx] if run["log"]]
        if not logs:
            continue
        min_len = min(map(len, logs))
        logs_trimmed = np.array([log[:min_len] for log in logs])
        mean_log = logs_trimmed.mean(axis=0)
        ax.plot(mean_log, label=name)

    ax.set_title(f"Setting {setting_idx + 1}")
    ax.set_xlabel("Generation / Evaluation")
    ax.set_ylabel("Training Profit")
    ax.grid(True)
    ax.legend()

for ax in axes[n_settings:]:
    ax.axis('off')

plt.tight_layout()
plt.show()

# --- Boxplots of Test Profits: All Settings in One Figure ---
n_cols = 2
n_rows = int(np.ceil(n_settings / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=False)
axes = axes.flatten()

for setting_idx in range(n_settings):
    ax = axes[setting_idx]
    data = [[run["info"]["profit"] for run in all_results[name][setting_idx]] for name in all_results]
    ax.boxplot(data, tick_labels=list(all_results.keys()))
    ax.set_title(f"Setting {setting_idx + 1}")
    ax.set_ylabel("Test Profit ($)")
    ax.grid(True)

for ax in axes[n_settings:]:
    ax.axis('off')

plt.tight_layout()
plt.show()

# --- Trade Count Bar Charts: All Settings in One Figure ---
n_cols = 2
n_rows = int(np.ceil(n_settings / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=False)
axes = axes.flatten()
x = np.arange(len(all_results))
width = 0.35
optimizer_names = list(all_results.keys())

for setting_idx in range(n_settings):
    ax = axes[setting_idx]
    avg_buys = [np.mean([len(run["info"]["buy_points"]) for run in all_results[name][setting_idx]]) for name in optimizer_names]
    avg_sells = [np.mean([len(run["info"]["sell_points"]) for run in all_results[name][setting_idx]]) for name in optimizer_names]

    ax.bar(x - width/2, avg_buys, width, label='Avg Buys', color='lightblue')
    ax.bar(x + width/2, avg_sells, width, label='Avg Sells', color='lightcoral')
    ax.set_xticks(x)
    ax.set_xticklabels(optimizer_names)
    ax.set_ylabel("Avg Number of Trades")
    ax.set_title(f"Setting {setting_idx + 1}")
    ax.legend()
    ax.grid(True, axis='y')

for ax in axes[n_settings:]:
    ax.axis('off')

plt.tight_layout()
plt.show()

# --- Trade Signal Subplots per Setting (2 Columns) ---
for setting_idx in range(n_settings):
    n_optimizers = len(all_results)
    n_cols = 2
    n_rows = int(np.ceil(n_optimizers / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))  # Removed sharex=True
    axes = axes.flatten()
    fig.suptitle(f"Trade Signals - Setting {setting_idx + 1}", fontsize=16)

    for ax, (name, runs) in zip(axes, all_results.items()):
        best_run = max(runs[setting_idx], key=lambda x: x["info"]["profit"])
        info = best_run["info"]

        ax.plot(info["price"], label="Price", alpha=0.6)
        ax.plot(info["high"], label="High (Composite)", linewidth=1.5)
        ax.plot(info["low"], label="Low (SMA)", linewidth=1.5)
        ax.scatter(info["buy_points"], info["price"][info["buy_points"]], marker='^', color='green', label='Buy', s=60)
        ax.scatter(info["sell_points"], info["price"][info["sell_points"]], marker='v', color='red', label='Sell', s=60)
        ax.set_title(f"{name} | Best Test Profit: ${info['profit']:.2f}")
        ax.grid(True)
        ax.legend()

    for ax in axes[len(all_results):]:
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# --- Combined Equity + Drawdown Subplots (2 Columns) ---
for setting_idx in range(n_settings):
    n_optimizers = len(all_results)
    n_cols = 2
    n_rows = int(np.ceil(n_optimizers / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()
    fig.suptitle(f"Equity & Drawdown - Setting {setting_idx + 1}", fontsize=16)

    for ax, (name, runs) in zip(axes, all_results.items()):
        best_run = max(runs[setting_idx], key=lambda x: x["info"]["profit"])
        info = best_run["info"]

        # Equity (left y-axis)
        ln1 = ax.plot(info["equity"], label="Equity", color="blue")
        ax.set_ylabel("Equity ($)", color="blue")
        ax.tick_params(axis='y', labelcolor="blue")
        ax.set_title(f"{name} | Profit: ${info['profit']:.2f}")
        ax.grid(True)

        # Drawdown (right y-axis)
        ax2 = ax.twinx()
        ln2 = ax2.plot(np.array(info["drawdown"]) * 100, label="Drawdown (%)", color="red", linestyle="--")
        ax2.set_ylabel("Drawdown (%)", color="red")
        ax2.tick_params(axis='y', labelcolor="red")

        # Combined legend
        lines = ln1 + ln2
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, loc="upper left")

    # Hide unused subplots
    for ax in axes[len(all_results):]:
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# --- Execution Time Comparison ---
bar_width = 0.13
x = np.arange(len(summary_df["Optimizer"].unique()))

plt.figure(figsize=(12, 6))
for i, cfg_id in enumerate(summary_df["Setting ID"].unique()):
    cfg_times = [
        summary_df[(summary_df["Optimizer"] == opt) & (summary_df["Setting ID"] == cfg_id)]["Exec Time (s)"].values[0]
        for opt in summary_df["Optimizer"].unique()
    ]
    plt.bar(x + i * bar_width, cfg_times, width=bar_width, label=f"Setting {cfg_id}")

plt.xticks(x + bar_width * (len(summary_df["Setting ID"].unique()) - 1) / 2, summary_df["Optimizer"].unique())
plt.ylabel("Avg Execution Time per Run (seconds)")
plt.title("Execution Time per Optimizer and Setting")
plt.legend(title="Setting ID")
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# --- Underperformance Rate per Optimizer per Setting ---
bar_width = 0.13
optimizers_list = summary_df["Optimizer"].unique()
settings_list = summary_df["Setting ID"].unique()
x = np.arange(len(optimizers_list))

plt.figure(figsize=(12, 6))
for i, setting_id in enumerate(settings_list):
    underperf = [
        summary_df[(summary_df["Optimizer"] == opt) & (summary_df["Setting ID"] == setting_id)]["Underperform (%)"].values[0]
        for opt in optimizers_list
    ]
    plt.bar(x + i * bar_width, underperf, width=bar_width, label=f"Setting {setting_id}")

plt.xticks(x + bar_width * (len(settings_list) - 1) / 2, optimizers_list)
plt.ylabel("Underperformance Rate (%)")
plt.title("Underperformance Rate per Optimizer per Setting")
plt.legend(title="Setting ID")
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# --- Most Frequent Timeframe per Optimizer per Setting ---
bar_width = 0.13
optimizers_list = summary_df["Optimizer"].unique()
settings_list = summary_df["Setting ID"].unique()
x = np.arange(len(optimizers_list))

plt.figure(figsize=(12, 6))
for i, setting_id in enumerate(settings_list):
    tf_hours = [
        summary_df[
            (summary_df["Optimizer"] == opt) & (summary_df["Setting ID"] == setting_id)
        ]["Timeframe (hours)"].values[0]
        for opt in optimizers_list
    ]
    plt.bar(x + i * bar_width, tf_hours, width=bar_width, label=f"Setting {setting_id}")

plt.xticks(x + bar_width * (len(settings_list) - 1) / 2, optimizers_list)
plt.ylabel("Timeframe (hours)")
plt.title("Most Frequent Timeframe per Optimizer per Setting")
plt.legend(title="Setting")
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# --- Best Parameters Summary ---
best_params_summary = []

for name in all_results:
    for setting_idx, setting_runs in enumerate(all_results[name]):
        # Find the best run (highest test profit)
        best_run = max(setting_runs, key=lambda x: x["info"]["profit"])
        best_params_summary.append({
            "Optimizer": name,
            "Setting ID": setting_idx + 1,
            "Best Test Profit": round(best_run["info"]["profit"], 2),
            "Best Params": best_run["params"],
            "Timeframe (hours)": best_run["timeframe"]
        })

# Convert to DataFrame
best_params_df = pd.DataFrame(best_params_summary)

print("\nBest Parameters per Optimizer and Setting:")
print(best_params_df.to_string(index=False))