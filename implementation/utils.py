import numpy as np
from filters import sma_filter, lma_filter, ema_filter, macd, wma
from config import initial_cash, fee

def pad(prices, window):
    shape = prices[1:window] - prices[0]
    padding = -np.flip(shape) + prices[0]
    return np.append(padding, prices)

def downsample(prices, interval):
    n = len(prices) // interval
    prices_trimmed = prices[:n * interval]
    return prices_trimmed.reshape(n, interval).mean(axis=1)

def quality(params, prices):
    w1, w2, w3, w4 = params[:4]
    d1, d2, d3 = map(int, params[4:7])
    alpha = params[7]
    d4, d5, d6 = map(int, params[8:11])
    d7 = int(params[11])

    high_components = []
    total_weight = w1 + w2 + w3 + w4
    if total_weight == 0:
        return 0

    high_components.append(w1 * wma(prices, d1, lma_filter(d1)))
    high_components.append(w2 * wma(prices, d2, sma_filter(d2)))
    high_components.append(w3 * wma(prices, d3, ema_filter(d3, alpha)))
    macd_line, _ = macd(prices, d4, d5, d6, alpha)
    high_components.append(w4 * macd_line)

    min_len = min(map(len, high_components))
    high = sum([h[-min_len:] for h in high_components]) / total_weight
    low = wma(prices, d7, sma_filter(d7))[-min_len:]
    price = prices[-min_len:]

    signal = high - low
    cross = np.sign(signal)
    triggers = np.convolve(cross, [1, -1], mode='valid')
    buy_points = np.where(triggers == 2)[0]
    sell_points = np.where(triggers == -2)[0]

    cash, btc = initial_cash, 0
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