import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from optimizer.abc_ import artificial_bee_colony
from optimizer.pso_ import particle_swarm
from optimizer.gwo_ import grey_wolf
from optimizer.woa_ import whale
from optimizer.fa_ import firefly
from optimizer.cs_ import cuckoo_search
from optimizer.sa_ import simulated_annealing

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

# --- Convergence Plot ---
for setting_idx in range(n_settings):
    plt.figure(figsize=(10, 6))
    for name in all_results:
        logs = [np.array(run["log"]) for run in all_results[name][setting_idx] if run["log"]]
        if not logs:
            continue
        min_len = min(map(len, logs))
        logs_trimmed = np.array([log[:min_len] for log in logs])
        mean_log = logs_trimmed.mean(axis=0)
        plt.plot(mean_log, label=name)
    plt.title(f"Average Convergence - Setting {setting_idx + 1}")
    plt.xlabel("Generation / Evaluation")
    plt.ylabel("Training Profit")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Boxplot of Test Profits ---
for setting_idx in range(n_settings):
    plt.figure(figsize=(10, 6))
    data = [[run["info"]["profit"] for run in all_results[name][setting_idx]] for name in all_results]
    plt.boxplot(data, tick_labels=all_results.keys())
    plt.title(f"Test Profit Distribution - Setting {setting_idx + 1}")
    plt.ylabel("Test Profit ($)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Trade Count Bar Charts ---
for setting_idx in range(n_settings):
    avg_buys = [np.mean([len(run["info"]["buy_points"]) for run in all_results[name][setting_idx]]) for name in all_results]
    avg_sells = [np.mean([len(run["info"]["sell_points"]) for run in all_results[name][setting_idx]]) for name in all_results]
    x = np.arange(len(all_results))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, avg_buys, width, label='Avg Buys', color='lightblue')
    plt.bar(x + width/2, avg_sells, width, label='Avg Sells', color='lightcoral')
    plt.xticks(x, all_results.keys())
    plt.ylabel("Avg Number of Trades")
    plt.title(f"Average Buys and Sells - Setting {setting_idx + 1}")
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

# --- Trade Signal Plots for Best Run ---
for setting_idx in range(n_settings):
    for name in all_results:
        best_run = max(all_results[name][setting_idx], key=lambda x: x["info"]["profit"])
        info = best_run["info"]

        plt.figure(figsize=(12, 4))
        plt.plot(info["price"], label="Price", alpha=0.6)
        plt.plot(info["high"], label="High (Composite)", linewidth=1.5)
        plt.plot(info["low"], label="Low (SMA)", linewidth=1.5)
        plt.scatter(info["buy_points"], info["price"][info["buy_points"]], marker='^', color='green', label='Buy', s=60)
        plt.scatter(info["sell_points"], info["price"][info["sell_points"]], marker='v', color='red', label='Sell', s=60)
        plt.title(f"{name} - Setting {setting_idx + 1} | Best Test Profit: ${info['profit']:.2f}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# --- Drawdown Plots for Best Runs ---
for setting_idx in range(n_settings):
    plt.figure(figsize=(14, 6 * len(all_results)))
    for idx, name in enumerate(all_results, 1):
        best_run = max(all_results[name][setting_idx], key=lambda x: x["info"]["profit"])
        info = best_run["info"]

        plt.subplot(len(all_results), 2, 2 * idx - 1)
        plt.plot(info["equity"], label="Equity Curve", color='blue')
        plt.title(f"{name} - Setting {setting_idx + 1} - Equity")
        plt.grid(True)
        plt.legend()

        plt.subplot(len(all_results), 2, 2 * idx)
        plt.plot(np.array(info["drawdown"]) * 100, label="Drawdown (%)", color='red')
        plt.title(f"{name} - Setting {setting_idx + 1} - Drawdown")
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
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