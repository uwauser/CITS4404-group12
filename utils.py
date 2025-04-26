import numpy as np
from filters import sma_filter, lma_filter, ema_filter, macd, wma
from config import initial_cash, fee

def pad(prices, window):
    """
    Pads the price series to ensure a consistent length for calculations.

    Args:
        prices (np.ndarray): The input price series.
        window (int): The window size for padding.

    Returns:
        np.ndarray: The padded price series.
    """
    shape = prices[1:window] - prices[0]
    padding = -np.flip(shape) + prices[0]
    return np.append(padding, prices)

def downsample(prices, interval):
    """
    Downsamples the price series by averaging over specified intervals.

    Args:
        prices (np.ndarray): The input price series.
        interval (int): The interval size for downsampling.

    Returns:
        np.ndarray: The downsampled price series.
    """
    n = len(prices) // interval
    prices_trimmed = prices[:n * interval]
    return prices_trimmed.reshape(n, interval).mean(axis=1)

def quality(params, prices, return_info=False):
    """
    Evaluates the quality of a trading strategy based on given parameters.

    Args:
        params (list): The parameters for the strategy.
        prices (np.ndarray): The input price series.
        return_info (bool, optional): Whether to return detailed information. Defaults to False.

    Returns:
        float or dict: The profit if `return_info` is False, otherwise a dictionary with detailed metrics.
    """
    w1, w2, w3, w4 = params[:4]
    d1, d2, d3 = map(int, params[4:7])
    alpha = params[7]
    d4, d5, d6 = map(int, params[8:11])
    d7 = int(params[11])

    total_weight = w1 + w2 + w3 + w4
    if total_weight == 0:
        return {
            "profit": 0,
            "price": [],
            "high": [],
            "low": [],
            "buy_points": [],
            "sell_points": [],
            "equity": [],
            "drawdown": []
        } if return_info else 0

    high_components = [
        w1 * wma(prices, d1, lma_filter(d1)),
        w2 * wma(prices, d2, sma_filter(d2)),
        w3 * wma(prices, d3, ema_filter(d3, alpha)),
        w4 * macd(prices, d4, d5, d6, alpha)[0]
    ]

    min_len = min(map(len, high_components))
    high = sum(h[-min_len:] for h in high_components) / total_weight
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

    if not return_info:
        return cash

    equity = compute_equity_curve(price, buy_points, sell_points)
    drawdown = compute_drawdown(equity)

    return {
        "profit": cash,
        "price": price,
        "high": high,
        "low": low,
        "buy_points": buy_points,
        "sell_points": sell_points,
        "equity": equity,
        "drawdown": drawdown
    }

def compute_equity_curve(price, buy_points, sell_points, fee=0.03):
    """
    Computes the equity curve for a trading strategy.

    Args:
        price (np.ndarray): The input price series.
        buy_points (np.ndarray): Indices of buy points.
        sell_points (np.ndarray): Indices of sell points.
        fee (float, optional): The transaction fee. Defaults to 0.03.

    Returns:
        np.ndarray: The equity curve.
    """
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
    """
    Computes the drawdown of an equity curve.

    Args:
        equity (np.ndarray): The equity curve.

    Returns:
        np.ndarray: The drawdown values.
    """
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    return drawdown