import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from filters import sma_filter, lma_filter, ema_filter, triangle_filter, gaussian_filter, step_filter, random_filter, polynomial_filter, polynomial_filter_centered, macd, wma
from utils import downsample

def profit(price, buy_points, sell_points, fee=0.01):
    cash = 1000
    btc = 0

    for buy, sell in zip(buy_points, sell_points):
        if buy < sell:
            buy_price = price[buy]
            sell_price = price[sell]

            btc = (1 - fee) * cash / buy_price
            cash = (1 - fee) * btc * sell_price
            btc = 0

    if btc > 0:
        cash = (1 - fee) * btc * price[-1]

    return round(cash, 2)

def plot_train_test_split(before_2020, after_2020, split_date):
    plt.figure(figsize=(14, 6))
    plt.plot(before_2020['date'], before_2020['close'], label='Training (Before 2020)', color='blue')
    plt.plot(after_2020['date'], after_2020['close'], label='Testing (From 2020)', color='green')
    plt.axvline(x=split_date, color='red', linestyle='--', label='Start of 2020 (Test Boundary)')
    plt.title("Dataset Coverage: Training vs Testing Periods")
    plt.xlabel("Date")
    plt.ylabel("Close Price (USD)")
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_filter_weights(window):

    cols = 4
    rows = int(np.ceil(len(filter_funcs) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(18, 4 * rows))
    axs = axs.flatten()

    for i, (name, func) in enumerate(filter_funcs.items()):
        weights = func(window)
        random_color = np.random.rand(3, )
        axs[i].bar(np.arange(window), weights, color=random_color)
        axs[i].set_title(name, fontsize=14, fontweight='bold', color=random_color)
        axs[i].set_xlabel('Kernel', fontsize=12)
        axs[i].set_ylabel('Weight', fontsize=12)
        axs[i].tick_params(axis='both', labelsize=10)

    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    plt.suptitle('Comparison of the weight profiles for implemented filters')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def test_and_plot_filter_against_SMA(prices, filter_func, name, window_high, window_low):
    high = wma(prices, window_high, filter_func(window_high))
    low = wma(prices, window_low, sma_filter(window_low))

    # Align lengths
    min_len = min(len(high), len(low))
    high = high[-min_len:]
    low = low[-min_len:]
    price = prices[-min_len:]

    # Detect crossovers
    signal = high - low
    cross = np.sign(signal)
    triggers = np.convolve(cross, [1, -1], mode='valid')
    buy_points = np.where(triggers == 2)[0]
    sell_points = np.where(triggers == -2)[0]

    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(price[-len(triggers):], label='Price', alpha=0.6, color='gray', marker='|')
    plt.plot(high[-len(triggers):], label=f'HIGH ({name})', linewidth=2)
    plt.plot(low[-len(triggers):], label='LOW (SMA)', linewidth=2)

    plt.scatter(buy_points, high[buy_points], marker='^', color='green', label='Buy', s=100)
    plt.scatter(sell_points, high[sell_points], marker='v', color='red', label='Sell', s=100)

    result = profit(price[-len(triggers):], buy_points, sell_points)
    summary_text = f"{name} vs SMA\nBuy: {len(buy_points)}\nSell: {len(sell_points)}\nMoney left: ${result:.2f}"

    plt.text(0.025, 0.95, summary_text, transform=plt.gca().transAxes,
         verticalalignment='top', horizontalalignment='left',
         fontsize=20, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.legend(fontsize=15)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(summary_text)

def test_and_plot_macd(macd, signal):
    # Align lengths
    min_len = min(len(macd), len(signal))
    macd = macd[-min_len:]
    signal = signal[-min_len:]
    price = prices[-min_len:]

    # Crossover logic
    macd_diff = macd - signal
    cross = np.sign(macd_diff)
    triggers = np.convolve(cross, [1, -1], mode='valid')
    buy_points = np.where(triggers == 2)[0]
    sell_points = np.where(triggers == -2)[0]

    result = profit(price[-len(triggers):], buy_points, sell_points)
    summary_text = f"MACD vs Signal\nBuy: {len(buy_points)}\nSell: {len(sell_points)}\nMoney left: ${result:.2f}"

    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(macd, label='MACD Line', linewidth=2)
    plt.plot(signal, label='Signal Line', linewidth=2)

    plt.scatter(buy_points, macd[buy_points], marker='^', color='green', label='Buy', s=100)
    plt.scatter(sell_points, macd[sell_points], marker='v', color='red', label='Sell', s=100)

    plt.text(0.025, 0.95, summary_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='left',
             fontsize=20, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.legend(fontsize=15)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(summary_text)


df = pd.read_csv("BTCUSD.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
split_date = pd.to_datetime('2020-01-01')
before_2020 = df[df['date'] < split_date]
after_2020 = df[df['date'] >= split_date]

alpha = 0.3
amplitude = 0.5
sigma = 1
timeframe = 168
poly_degrees = [1, 2]
prices = downsample(before_2020['close'].dropna().values, timeframe)

filter_funcs = {
    'SMA': sma_filter,
    'LMA': lma_filter,
    f'EMA (α={alpha})': lambda window: ema_filter(window, alpha),
    'Triangle': triangle_filter,
    f'Gaussian (sigma={sigma})': lambda window: gaussian_filter(window, sigma),
    f'Step (amplitude={amplitude})': lambda window: step_filter(window, amplitude),
    'Random': random_filter
}

for degree in poly_degrees:
    filter_funcs[f'Poly (degree={degree})'] = lambda window, d=degree: polynomial_filter(window, degree=d)
    filter_funcs[f'Poly Centered (degree={degree})'] = lambda window, d=degree: polynomial_filter_centered(window, degree=d)

macd, signal = macd(prices, fast=12, slow=26, signal=9)

plot_train_test_split(before_2020, after_2020, split_date)
plot_filter_weights(window=9)

for name, func in filter_funcs.items():
    test_and_plot_filter_against_SMA(prices, func, name, window_high=10, window_low=50)

test_and_plot_macd(macd, signal)
