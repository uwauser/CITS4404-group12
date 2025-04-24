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

from utils import quality, downsample, compute_equity_curve, compute_drawdown
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
        print(f"\nRunning {name} - Config {idx+1}")
        results = []
        start_time = time.time()

        for run in range(n_runs):
            log = []
            best_params, train_profit, tf = optimizer(train_prices, log=log, setting=setting)
            test_ds_prices = downsample(test_prices, tf)
            test_profit, buys, sells = quality(best_params[:-1], test_ds_prices, return_trades=True)

            results.append({
                "train_profit": train_profit,
                "test_profit": test_profit,
                "params": best_params,
                "timeframe": tf,
                "log": log,
                "buys": buys,
                "sells": sells
            })

        exec_time = (time.time() - start_time) / n_runs

        all_results[name].append(results)

        # --- Metrics Calculation ---
        train = [r["train_profit"] for r in results]
        test = [r["test_profit"] for r in results]
        buys = [r["buys"] for r in results]
        sells = [r["sells"] for r in results]
        convergence = [len(r["log"]) for r in results]
        timeframes = [r["timeframe"] for r in results]
        underperform = [p <= 1000 for p in test]

        drawdowns = []
        for r in results:
            tf = r["timeframe"]
            test_ds_prices = downsample(test_prices, tf)

            w1, w2, w3, w4 = r["params"][:4]
            d1, d2, d3 = map(int, r["params"][4:7])
            alpha = r["params"][7]
            d4, d5, d6 = map(int, r["params"][8:11])
            d7 = int(r["params"][11])

            from filters import lma_filter, sma_filter, ema_filter, macd, wma
            total_weight = w1 + w2 + w3 + w4
            if total_weight == 0:
                continue

            high_components = [
                w1 * wma(test_ds_prices, d1, lma_filter(d1)),
                w2 * wma(test_ds_prices, d2, sma_filter(d2)),
                w3 * wma(test_ds_prices, d3, ema_filter(d3, alpha)),
                w4 * macd(test_ds_prices, d4, d5, d6, alpha)[0]
            ]

            min_len = min(map(len, high_components))
            high = sum(h[-min_len:] for h in high_components) / total_weight
            low = wma(test_ds_prices, d7, sma_filter(d7))[-min_len:]
            price = test_ds_prices[-min_len:]

            signal = high - low
            cross = np.sign(signal)
            triggers = np.convolve(cross, [1, -1], mode='valid')
            buy_points = np.where(triggers == 2)[0]
            sell_points = np.where(triggers == -2)[0]

            equity = compute_equity_curve(price, buy_points, sell_points)
            drawdown = compute_drawdown(equity)
            drawdowns.append(np.mean(drawdown) * 100)

        # --- Append Summary Row ---
        summary.append({
            "Optimizer": name,
            "Config ID": idx + 1,
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

# --- Print Summary ---
summary_df = pd.DataFrame(summary)
print("\nSummary Table:")
print(summary_df.to_string(index=False))

# --- Average Convergence Plot: All Optimizers per Config ---
n_configs = len(next(iter(all_results.values())))

for cfg_idx in range(n_configs):
    plt.figure(figsize=(10, 6))
    for name in all_results:
        runs = all_results[name][cfg_idx]
        logs = [np.array(run["log"]) for run in runs if run["log"]]
        if not logs:
            continue
        min_len = min(map(len, logs))
        logs_trimmed = np.array([log[:min_len] for log in logs])
        mean_log = logs_trimmed.mean(axis=0)
        plt.plot(mean_log, label=name)
    plt.title(f"Average Convergence - Config {cfg_idx + 1}")
    plt.xlabel("Generation / Evaluation")
    plt.ylabel("Training Profit")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Boxplot of Test Profits per Config Across Optimizers ---
for cfg_idx in range(n_configs):
    plt.figure(figsize=(10, 6))
    data = [  # Collect test profits from all optimizers for current config
        [run["test_profit"] for run in all_results[name][cfg_idx]]
        for name in all_results
    ]
    plt.boxplot(data, tick_labels=all_results.keys())
    plt.title(f"Test Profit Distribution - Config {cfg_idx + 1}")
    plt.ylabel("Test Profit ($)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Trade Count Distribution per Config Across Optimizers ---
for cfg_idx in range(n_configs):
    avg_buys = [
        np.mean([run["buys"] for run in all_results[name][cfg_idx]])
        for name in all_results
    ]
    avg_sells = [
        np.mean([run["sells"] for run in all_results[name][cfg_idx]])
        for name in all_results
    ]
    x = np.arange(len(all_results))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, avg_buys, width, label='Avg Buys', color='lightblue')
    plt.bar(x + width/2, avg_sells, width, label='Avg Sells', color='lightcoral')
    plt.xticks(x, all_results.keys())
    plt.ylabel("Avg Number of Trades")
    plt.title(f"Average Buys and Sells - Config {cfg_idx + 1}")
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

# --- Best Trade Plots per Optimizer for Each Config ---
for cfg_idx in range(n_configs):
    for name in all_results:
        best_run = max(all_results[name][cfg_idx], key=lambda x: x["test_profit"])
        tf = best_run["timeframe"]
        test_ds_prices = downsample(test_prices, tf)

        w1, w2, w3, w4 = best_run["params"][:4]
        d1, d2, d3 = map(int, best_run["params"][4:7])
        alpha = best_run["params"][7]
        d4, d5, d6 = map(int, best_run["params"][8:11])
        d7 = int(best_run["params"][11])

        from filters import lma_filter, sma_filter, ema_filter, macd, wma

        total_weight = w1 + w2 + w3 + w4
        if total_weight == 0:
            continue

        high_components = [
            w1 * wma(test_ds_prices, d1, lma_filter(d1)),
            w2 * wma(test_ds_prices, d2, sma_filter(d2)),
            w3 * wma(test_ds_prices, d3, ema_filter(d3, alpha)),
            w4 * macd(test_ds_prices, d4, d5, d6, alpha)[0]
        ]

        min_len = min(map(len, high_components))
        high = sum(h[-min_len:] for h in high_components) / total_weight
        low = wma(test_ds_prices, d7, sma_filter(d7))[-min_len:]
        price = test_ds_prices[-min_len:]

        signal = high - low
        cross = np.sign(signal)
        triggers = np.convolve(cross, [1, -1], mode='valid')
        buy_points = np.where(triggers == 2)[0]
        sell_points = np.where(triggers == -2)[0]

        plt.figure(figsize=(12, 4))
        plt.plot(price, label="Price", alpha=0.6)
        plt.plot(high, label="High (Composite)", linewidth=1.5)
        plt.plot(low, label="Low (SMA)", linewidth=1.5)
        plt.scatter(buy_points, price[buy_points], marker='^', color='green', label='Buy', s=60)
        plt.scatter(sell_points, price[sell_points], marker='v', color='red', label='Sell', s=60)
        plt.title(f"{name} - Config {cfg_idx + 1} | Best Test Profit: ${best_run['test_profit']:.2f}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# --- Drawdown Plots for Best Run of Each Optimizer (Subplots per Config) ---
for cfg_idx in range(n_configs):
    plt.figure(figsize=(14, 6 * len(all_results)))  # One config, all optimizers vertically stacked

    for idx, name in enumerate(all_results, 1):
        best_run = max(all_results[name][cfg_idx], key=lambda x: x["test_profit"])
        tf = best_run["timeframe"]
        test_ds_prices = downsample(test_prices, tf)

        w1, w2, w3, w4 = best_run["params"][:4]
        d1, d2, d3 = map(int, best_run["params"][4:7])
        alpha = best_run["params"][7]
        d4, d5, d6 = map(int, best_run["params"][8:11])
        d7 = int(best_run["params"][11])

        from filters import lma_filter, sma_filter, ema_filter, macd, wma

        total_weight = w1 + w2 + w3 + w4
        if total_weight == 0:
            continue

        high_components = [
            w1 * wma(test_ds_prices, d1, lma_filter(d1)),
            w2 * wma(test_ds_prices, d2, sma_filter(d2)),
            w3 * wma(test_ds_prices, d3, ema_filter(d3, alpha)),
            w4 * macd(test_ds_prices, d4, d5, d6, alpha)[0]
        ]

        min_len = min(map(len, high_components))
        high = sum([h[-min_len:] for h in high_components]) / total_weight
        low = wma(test_ds_prices, d7, sma_filter(d7))[-min_len:]
        price = test_ds_prices[-min_len:]

        signal = high - low
        cross = np.sign(signal)
        triggers = np.convolve(cross, [1, -1], mode='valid')
        buy_points = np.where(triggers == 2)[0]
        sell_points = np.where(triggers == -2)[0]

        equity = compute_equity_curve(price, buy_points, sell_points)
        drawdown = compute_drawdown(equity)

        plt.subplot(len(all_results), 2, 2 * idx - 1)
        plt.plot(equity, label="Equity Curve", color='blue')
        plt.title(f"{name} - Config {cfg_idx + 1} - Equity")
        plt.grid(True)
        plt.legend()

        plt.subplot(len(all_results), 2, 2 * idx)
        plt.plot(drawdown * 100, label="Drawdown (%)", color='red')
        plt.title(f"{name} - Config {cfg_idx + 1} - Drawdown")
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.show()

# --- Execution Time Comparison Plot ---
exec_times = [np.mean([res["exec_time"] for res in results[name]]) for name in optimizers]

plt.figure(figsize=(10, 6))
plt.bar(optimizers.keys(), exec_times, color='steelblue')
plt.ylabel("Average Execution Time per Run (seconds)")
plt.title("Execution Time Comparison per Optimizer")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()