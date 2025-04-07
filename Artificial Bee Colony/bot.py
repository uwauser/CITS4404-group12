import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load historical bitcoin data
file_path = "Bitcoin_4_7_2024-4_7_2025_historical_data_coinmarketcap.csv"       # 1year
#file_path = "Bitcoin_4_1_2020-4_1_2022_historical_data_coinmarketcap.csv"      # 2 years
btc_data = pd.read_csv(file_path, delimiter=';')
btc_data['timeOpen'] = pd.to_datetime(btc_data['timeOpen'])
btc_data = btc_data.sort_values('timeOpen').reset_index(drop=True)
btc_data = btc_data[['timeOpen', 'close']]
btc_data.rename(columns={'timeOpen': 'Date', 'close': 'Close'}, inplace=True)
btc_data['Close'] = pd.to_numeric(btc_data['Close'], errors='coerce')
btc_data.dropna(inplace=True)

# Extract price values
prices = btc_data['Close'].values

# SMA computation
def SMA(data, window):
    return np.convolve(data, np.ones(window) / window, mode='valid')

# Evaluation function with enforced short < long logic
def evaluate_bot(params):
    short_window = int(abs(params[0])) + 1
    gap = int(abs(params[1])) + 1
    long_window = short_window + gap

    if long_window > len(prices):
        return 0  # invalid config

    sma_short = SMA(prices, short_window)
    sma_long = SMA(prices, long_window)

    min_len = min(len(sma_short), len(sma_long))
    sma_short = sma_short[-min_len:]
    sma_long = sma_long[-min_len:]
    price_effective = prices[-min_len:]

    diff = sma_short - sma_long
    signals = np.sign(diff)
    triggers = np.convolve(signals, [1, -1], mode='valid')
    price_effective = price_effective[-len(triggers):]
    date_effective = btc_data['Date'].values[-len(triggers):]

    state = 'cash'
    cash, btc = 1000, 0
    buy_signals = []
    sell_signals = []

    for i in range(len(triggers)):
        if triggers[i] > 0.5 and state == 'cash':
            btc = 0.97 * cash / price_effective[i]
            cash = 0
            state = 'btc'
            buy_signals.append((date_effective[i], price_effective[i]))
        elif triggers[i] < -0.5 and state == 'btc':
            cash = 0.97 * btc * price_effective[i]
            btc = 0
            state = 'cash'
            sell_signals.append((date_effective[i], price_effective[i]))

    if state == 'btc':
        cash = 0.97 * btc * price_effective[-1]
        sell_signals.append((date_effective[-1], price_effective[-1]))

    evaluate_bot.last_trades = {'buy': buy_signals, 'sell': sell_signals}

    return cash

# Artificial Bee Colony Optimization
def abc_optimize(fitness_func, n_bees=20, max_iter=50, dim=2, limit=10):
    lb, ub = 1, 50  # SMA/gap range
    foods = np.random.uniform(lb, ub, (n_bees, dim))
    fitness = np.array([fitness_func(sol) for sol in foods])
    trial = np.zeros(n_bees)

    for gen in range(max_iter):
        # Employed bees
        for i in range(n_bees):
            k = np.random.randint(n_bees)
            while k == i:
                k = np.random.randint(n_bees)
            phi = np.random.uniform(-1, 1, dim)
            new_sol = foods[i] + phi * (foods[i] - foods[k])
            new_sol = np.clip(new_sol, lb, ub)
            new_fit = fitness_func(new_sol)
            if new_fit > fitness[i]:
                foods[i] = new_sol
                fitness[i] = new_fit
                trial[i] = 0
            else:
                trial[i] += 1

        # Onlooker bees
        prob = fitness / np.sum(fitness)
        for i in range(n_bees):
            if np.random.rand() < prob[i]:
                k = np.random.randint(n_bees)
                while k == i:
                    k = np.random.randint(n_bees)
                phi = np.random.uniform(-1, 1, dim)
                new_sol = foods[i] + phi * (foods[i] - foods[k])
                new_sol = np.clip(new_sol, lb, ub)
                new_fit = fitness_func(new_sol)
                if new_fit > fitness[i]:
                    foods[i] = new_sol
                    fitness[i] = new_fit
                    trial[i] = 0
                else:
                    trial[i] += 1

        # Scout bees
        for i in range(n_bees):
            if trial[i] > limit:
                foods[i] = np.random.uniform(lb, ub, dim)
                fitness[i] = fitness_func(foods[i])
                trial[i] = 0

    best_idx = np.argmax(fitness)
    return foods[best_idx], fitness[best_idx]

# Run optimization on data
best_params, best_profit = abc_optimize(evaluate_bot)

# Extract windows
short_sma = int(abs(best_params[0])) + 1
gap = int(abs(best_params[1])) + 1
long_sma = short_sma + gap

print(f"Best SMA windows: Short = {short_sma}, Long = {long_sma}")
print(f"Final portfolio value: ${best_profit:.2f}")

# Plot price with optimized SMAs
sma_short_series = SMA(prices, short_sma)
sma_long_series = SMA(prices, long_sma)

min_len = min(len(sma_short_series), len(sma_long_series))
plot_dates = btc_data['Date'].values[-min_len:]

sma_short_series = sma_short_series[-min_len:]
sma_long_series = sma_long_series[-min_len:]
price_plot = prices[-min_len:]

plt.figure(figsize=(12, 6))
plt.plot(plot_dates, price_plot, label='BTC Price', linewidth=1.5)
plt.plot(plot_dates, sma_short_series, label=f'SMA Short ({short_sma})', linestyle='--')
plt.plot(plot_dates, sma_long_series, label=f'SMA Long ({long_sma})', linestyle='--')
plt.title("BTC Price with optimized SMAs")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
