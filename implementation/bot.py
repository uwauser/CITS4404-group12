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
results = {name: [] for name in optimizers}

# --- Run Experiments ---
for name, optimizer in optimizers.items():
    print(f"\nRunning optimizer: {name}")
    start_time = time.time()

    for run in range(n_runs):
        print(f"Run {run+1}/{n_runs}")
        log = []
        best_params, train_profit, tf = optimizer(train_prices, log=log)
        test_ds_prices = downsample(test_prices, tf)
        test_profit, buys, sells = quality(best_params[:-1], test_ds_prices, return_trades=True)

        results[name].append({
            "train_profit": train_profit,
            "test_profit": test_profit,
            "params": best_params,
            "timeframe": tf,
            "log": log,
            "buys": buys,
            "sells": sells
        })

    end_time = time.time()
    total_time = end_time - start_time
    for res in results[name]:
        res["exec_time"] = total_time / n_runs

# --- Average Convergence Plot over 10 Runs ---
plt.figure(figsize=(10, 6))
for name in optimizers:
    logs = [np.array(run["log"]) for run in results[name]]
    min_len = min(map(len, logs))
    logs_trimmed = np.array([log[:min_len] for log in logs])
    mean_log = logs_trimmed.mean(axis=0)
    plt.plot(mean_log, label=name)

plt.title("Average Convergence (Over 10 Runs)")
plt.xlabel("Generation / Evaluation")
plt.ylabel("Training Profit")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Boxplot of Test Profits ---
plt.figure(figsize=(10, 6))
plt.boxplot([[res["test_profit"] for res in results[name]] for name in optimizers], tick_labels=optimizers.keys())
plt.title("Test Profit Distribution per Optimizer")
plt.ylabel("Test Profit ($)")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Trade Count Distribution Bar Chart ---
avg_buys = [np.mean([res["buys"] for res in results[name]]) for name in optimizers]
avg_sells = [np.mean([res["sells"] for res in results[name]]) for name in optimizers]
x = np.arange(len(optimizers))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, avg_buys, width, label='Avg Buys', color='lightblue')
plt.bar(x + width/2, avg_sells, width, label='Avg Sells', color='lightcoral')
plt.xticks(x, optimizers.keys())
plt.ylabel("Average Number of Trades")
plt.title("Average Buy and Sell Counts per Optimizer")
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# --- Summary Table ---
summary = {
    "Optimizer": [],
    "Avg Train": [],
    "Avg": [],
    "Max": [],
    "Min": [],
    "Std": [],
    "Avg Buys": [],
    "Avg Sells": [],
    "Avg Convergence": [],
    "Timeframe (hours)": [],
    "Underperform (%)": [],
    "Avg Drawdown (%)": [],
    "Exec Time (s)": []
}

for name in optimizers:
    train = [res["train_profit"] for res in results[name]]
    test = [res["test_profit"] for res in results[name]]
    buys = [res["buys"] for res in results[name]]
    sells = [res["sells"] for res in results[name]]
    convergence = [len(res["log"]) for res in results[name]]
    timeframes = [res["timeframe"] for res in results[name]]
    underperform  = [p <= 1000 for p in test]

    drawdowns = []
    for res in results[name]:
        tf = res["timeframe"]
        test_ds_prices = downsample(test_prices, tf)

        # Recompute for equity
        w1, w2, w3, w4 = res["params"][:4]
        d1, d2, d3 = map(int, res["params"][4:7])
        alpha = res["params"][7]
        d4, d5, d6 = map(int, res["params"][8:11])
        d7 = int(res["params"][11])

        from filters import lma_filter, sma_filter, ema_filter, macd, wma

        high_components = []
        total_weight = w1 + w2 + w3 + w4
        if total_weight == 0:
            continue

        high_components.append(w1 * wma(test_ds_prices, d1, lma_filter(d1)))
        high_components.append(w2 * wma(test_ds_prices, d2, sma_filter(d2)))
        high_components.append(w3 * wma(test_ds_prices, d3, ema_filter(d3, alpha)))
        macd_line, _ = macd(test_ds_prices, d4, d5, d6, alpha)
        high_components.append(w4 * macd_line)

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
        avg_dd = np.mean(drawdown) * 100  # Convert to %
        drawdowns.append(avg_dd)

    exec_times = [res["exec_time"] for res in results[name]]

    summary["Optimizer"].append(name)
    summary["Avg Train"].append(round(np.mean(train), 2))
    summary["Avg"].append(round(np.mean(test), 2))
    summary["Max"].append(round(np.max(test), 2))
    summary["Min"].append(round(np.min(test), 2))
    summary["Std"].append(round(np.std(test), 2))
    summary["Avg Buys"].append(round(np.mean(buys), 2))
    summary["Avg Sells"].append(round(np.mean(sells), 2))
    summary["Avg Convergence"].append(round(np.mean(convergence), 2))
    summary["Timeframe (hours)"].append(max(set(timeframes), key=timeframes.count))
    summary["Underperform (%)"].append(round(100 * np.mean(underperform), 1))
    summary["Avg Drawdown (%)"].append(round(np.mean(drawdowns), 2))
    summary["Exec Time (s)"].append(round(np.mean(exec_times), 2))

summary_df = pd.DataFrame(summary)
print("\nSummary Table:")
print(summary_df.to_string(index=False))

# --- Trade Plots for Best Test Run from Each Optimizer ---
plt.figure(figsize=(14, 6 * len(optimizers)))

for idx, name in enumerate(optimizers, 1):
    best_run = max(results[name], key=lambda x: x["test_profit"])
    tf = best_run["timeframe"]
    test_ds_prices = downsample(test_prices, tf)

    # Recompute for clarity (same as in quality)
    w1, w2, w3, w4 = best_run["params"][:4]
    d1, d2, d3 = map(int, best_run["params"][4:7])
    alpha = best_run["params"][7]
    d4, d5, d6 = map(int, best_run["params"][8:11])
    d7 = int(best_run["params"][11])

    from filters import lma_filter, sma_filter, ema_filter, macd, wma

    high_components = []
    total_weight = w1 + w2 + w3 + w4
    if total_weight == 0:
        continue

    high_components.append(w1 * wma(test_ds_prices, d1, lma_filter(d1)))
    high_components.append(w2 * wma(test_ds_prices, d2, sma_filter(d2)))
    high_components.append(w3 * wma(test_ds_prices, d3, ema_filter(d3, alpha)))
    macd_line, _ = macd(test_ds_prices, d4, d5, d6, alpha)
    high_components.append(w4 * macd_line)

    min_len = min(map(len, high_components))
    high = sum([h[-min_len:] for h in high_components]) / total_weight
    low = wma(test_ds_prices, d7, sma_filter(d7))[-min_len:]
    price = test_ds_prices[-min_len:]

    signal = high - low
    cross = np.sign(signal)
    triggers = np.convolve(cross, [1, -1], mode='valid')
    buy_points = np.where(triggers == 2)[0]
    sell_points = np.where(triggers == -2)[0]

    plt.subplot(len(optimizers), 1, idx)
    plt.plot(price, label="Price", alpha=0.6)
    plt.plot(high, label="High (Composite)", linewidth=1.5)
    plt.plot(low, label="Low (SMA)", linewidth=1.5)
    plt.scatter(buy_points, price[buy_points], marker='^', color='green', label='Buy', s=60)
    plt.scatter(sell_points, price[sell_points], marker='v', color='red', label='Sell', s=60)
    plt.title(f"{name} - Best Run | Test Profit: ${best_run['test_profit']:.2f}")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# --- Drawdown Plots for Best Test Run of Each Optimizer ---
plt.figure(figsize=(14, 6 * len(optimizers)))

for idx, name in enumerate(optimizers, 1):
    best_run = max(results[name], key=lambda x: x["test_profit"])
    tf = best_run["timeframe"]
    test_ds_prices = downsample(test_prices, tf)

    # Recompute buy/sell points for equity curve
    w1, w2, w3, w4 = best_run["params"][:4]
    d1, d2, d3 = map(int, best_run["params"][4:7])
    alpha = best_run["params"][7]
    d4, d5, d6 = map(int, best_run["params"][8:11])
    d7 = int(best_run["params"][11])

    from filters import lma_filter, sma_filter, ema_filter, macd, wma

    high_components = []
    total_weight = w1 + w2 + w3 + w4
    if total_weight == 0:
        continue

    high_components.append(w1 * wma(test_ds_prices, d1, lma_filter(d1)))
    high_components.append(w2 * wma(test_ds_prices, d2, sma_filter(d2)))
    high_components.append(w3 * wma(test_ds_prices, d3, ema_filter(d3, alpha)))
    macd_line, _ = macd(test_ds_prices, d4, d5, d6, alpha)
    high_components.append(w4 * macd_line)

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

    plt.subplot(len(optimizers), 2, 2 * idx - 1)
    plt.plot(equity, label="Equity Curve", color='blue')
    plt.title(f"{name} - Equity Curve")
    plt.grid(True)
    plt.legend()

    plt.subplot(len(optimizers), 2, 2 * idx)
    plt.plot(drawdown * 100, label="Drawdown %", color='red')
    plt.title(f"{name} - Drawdown")
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