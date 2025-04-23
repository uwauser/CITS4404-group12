import pandas as pd
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

# --- Run ABC Optimization on Training Data ---
abc_params, abc_profit, abc_tf = artificial_bee_colony(train_prices)
print("\n--- ABC Optimizer Results ---")
print("Best Params:", abc_params)
print("Training Profit:", abc_profit)

# --- Evaluate ABC on Test Data ---
test_ds_prices = downsample(test_prices, abc_tf)
abc_test_profit = quality(abc_params[:-1], test_ds_prices)
print("Test Profit:", abc_test_profit)

# --- Run PSO Optimization on Training Data ---
pso_params, pso_profit, pso_tf = particle_swarm(train_prices)
print("\n--- PSO Optimizer Results ---")
print("Best Params:", pso_params)
print("Training Profit:", pso_profit)

# --- Evaluate PSO on Test Data ---
test_ds_prices = downsample(test_prices, pso_tf)
pso_test_profit = quality(pso_params[:-1], test_ds_prices)
print("Test Profit:", pso_test_profit)

# --- Run GWO Optimization on Training Data ---
gwo_params, gwo_profit, gwo_tf = grey_wolf(train_prices)
print("\n--- GWO Optimizer Results ---")
print("Best Params:", gwo_params)
print("Training Profit:", gwo_profit)

# --- Evaluate GWO on Test Data ---
test_ds_prices = downsample(test_prices, gwo_tf)
gwo_test_profit = quality(gwo_params[:-1], test_ds_prices)
print("Test Profit:", gwo_test_profit)

# --- Run WOA Optimization on Training Data ---
woa_params, woa_profit, woa_tf = whale(train_prices)
print("\n--- WOA Optimizer Results ---")
print("Best Params:", woa_params)
print("Training Profit:", woa_profit)

# --- Evaluate WOA on Test Data ---
test_ds_prices = downsample(test_prices, woa_tf)
woa_test_profit = quality(woa_params[:-1], test_ds_prices)
print("Test Profit:", woa_test_profit)

# --- Run FA Optimization on Training Data ---
fa_params, fa_profit, fa_tf = firefly(train_prices)
print("\n--- FA Optimizer Results ---")
print("Best Params:", fa_params)
print("Training Profit:", fa_profit)

# --- Evaluate FA on Test Data ---
test_ds_prices = downsample(test_prices, fa_tf)
fa_test_profit = quality(fa_params[:-1], test_ds_prices)
print("Test Profit:", fa_test_profit)

# --- Run CS Optimization on Training Data ---
cs_params, cs_profit, cs_tf = cuckoo_search(train_prices)
print("\n--- CS Optimizer Results ---")
print("Best Params:", cs_params)
print("Training Profit:", cs_profit)

# --- Evaluate CS on Test Data ---
test_ds_prices = downsample(test_prices, cs_tf)
cs_test_profit = quality(cs_params[:-1], test_ds_prices)
print("Test Profit:", cs_test_profit)

# --- Run SA Optimization on Training Data ---
sa_params, sa_profit, sa_tf = simulated_annealing(train_prices)
print("\n--- SA Optimizer Results ---")
print("Best Params:", sa_params)
print("Training Profit:", sa_profit)

# --- Evaluate SA on Test Data ---
test_ds_prices = downsample(test_prices, sa_tf)
sa_test_profit = quality(sa_params[:-1], test_ds_prices)
print("Test Profit:", sa_test_profit)

def summary_table():
    data = {
        "Optimizer": ["ABC", "PSO", "GWO", "WOA", "FA", "CS", "SA"],
        "Train Profit": [abc_profit, pso_profit, gwo_profit, woa_profit, fa_profit, cs_profit, sa_profit],
        "Test Profit": [abc_test_profit, pso_test_profit, gwo_test_profit, woa_test_profit, fa_test_profit,
                        cs_test_profit, sa_test_profit],
    }
    df = pd.DataFrame(data)
    print("\n=== Summary of Optimizer Performance ===")
    print(df.to_string(index=False))
    return df

# Call the summary at the end
summary_table()

def plot_test_profits():
    optimizers = ["ABC", "PSO", "GWO", "WOA", "FA", "CS"]
    test_profits = [abc_test_profit, pso_test_profit, gwo_test_profit, woa_test_profit, fa_test_profit, cs_test_profit]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(optimizers, test_profits)
    plt.title("Test Profit Comparison")
    plt.ylabel("Profit ($)")
    plt.xlabel("Optimizer")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Annotate bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 20, f"${yval:.0f}", ha='center', va='bottom')

    plt.show()

# Call it at the end of bot.py
plot_test_profits()