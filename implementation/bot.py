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
test_profits_all = {name: [] for name in optimizers}

# --- Run Experiments ---
for name, optimizer in optimizers.items():
    print(f"\nRunning optimizer: {name}")
    for run in range(n_runs):
        print(f"Run {run+1}/{n_runs}")
        log = []
        best_params, train_profit, tf = optimizer(train_prices, log=log)
        test_ds_prices = downsample(test_prices, tf)
        test_profit = quality(best_params[:-1], test_ds_prices)
        test_profits_all[name].append(test_profit)
        results[name].append({
            "train_profit": train_profit,
            "test_profit": test_profit,
            "params": best_params,
            "timeframe": tf,
            "log": log
        })

# --- Convergence Plot (first run only) ---
plt.figure(figsize=(10, 6))
for name in optimizers:
    first_log = results[name][0]["log"]
    plt.plot(first_log, label=name)
plt.title("Convergence (First Run Only)")
plt.xlabel("Generation / Evaluation")
plt.ylabel("Training Profit")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Boxplot of Test Profits ---
plt.figure(figsize=(10, 6))
plt.boxplot([test_profits_all[name] for name in optimizers], tick_labels=optimizers.keys())
plt.title("Test Profit Distribution per Optimizer")
plt.ylabel("Test Profit ($)")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Summary Table ---
summary = {
    "Optimizer": [],
    "Best": [],
    "Average": [],
    "Std Dev": [],
}

for name in optimizers:
    profits = test_profits_all[name]
    summary["Optimizer"].append(name)
    summary["Best"].append(round(np.max(profits), 2))
    summary["Average"].append(round(np.mean(profits), 2))
    summary["Std Dev"].append(round(np.std(profits), 2))

summary_df = pd.DataFrame(summary)
print("\nSummary Table:")
print(summary_df.to_string(index=False))
