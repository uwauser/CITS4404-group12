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

def quality(params, prices, return_trades=False):
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

    cash = round(cash, 2)

    return (cash, len(buy_points), len(sell_points)) if return_trades else cash

def compute_equity_curve(price, buy_points, sell_points, fee=0.03):
    cash, btc = 1000, 0
    equity_curve = np.zeros_like(price, dtype=float)

    holding = 'cash'
    next_trade = 0

    trades = sorted([(i, 'buy') for i in buy_points] + [(i, 'sell') for i in sell_points], key=lambda x: x[0])

    for i in range(len(price)):
        while next_trade < len(trades) and trades[next_trade][0] == i:
            if trades[next_trade][1] == 'buy' and holding == 'cash':
                btc = (1 - fee) * cash / price[i]
                cash = 0
                holding = 'btc'
            elif trades[next_trade][1] == 'sell' and holding == 'btc':
                cash = (1 - fee) * btc * price[i]
                btc = 0
                holding = 'cash'
            next_trade += 1

        equity_curve[i] = cash if holding == 'cash' else btc * price[i]

    return equity_curve

def compute_drawdown(equity):
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    return drawdown