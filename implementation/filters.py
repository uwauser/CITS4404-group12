import numpy as np

def pad(prices, window):
    shape = prices[1:window] - prices[0]
    padding = -np.flip(shape) + prices[0]
    return np.append(padding, prices)

def sma_filter(window):
    return np.ones(window) / window

def lma_filter(window):
    kernel = np.arange(window)
    weights = (2 / (window + 1)) * (1 - kernel / window)
    return weights / np.sum(weights)

def ema_filter(window, alpha):
    if alpha is None:
        alpha = 2 / (window + 1)
    kernel = np.arange(window)
    weights = alpha * (1 - alpha) ** kernel
    return weights / np.sum(weights)

def triangle_filter(window):
    if window % 2 == 0:
        half = np.arange(1, window // 2 + 1)
        weights = np.concatenate([half, half[::-1]])
    else:
        half = np.arange(1, (window + 1) // 2 + 1)
        weights = np.concatenate([half, half[-2::-1]])
    return weights / np.sum(weights)

def gaussian_filter(window, sigma=None):
    if sigma is None:
        sigma = window / 6
    kernel = np.arange(window)
    center = (window - 1) / 2
    weights = np.exp(-0.5 * ((kernel - center) / sigma) ** 2)
    return weights / np.sum(weights)

def step_filter(window, amplitude=2.0):
    weights = np.ones(window)
    weights = weights / np.sum(weights)
    weights *= amplitude
    return weights

def random_filter(window, seed=42):
    np.random.seed(seed)
    weights = np.random.rand(window)
    return weights / np.sum(weights)

def polynomial_filter(window, degree=1):
    kernel = np.arange(window)
    weights = kernel ** degree
    return weights / np.sum(weights)

def polynomial_filter_centered(window, degree=2):
    kernel = np.arange(window)
    center = (window - 1) / 2
    weights = np.abs(kernel - center) ** degree
    return weights / np.sum(weights)

def wma(prices, window, kernel):
    return np.convolve(pad(prices, window), kernel, mode='valid')

def macd(prices, fast, slow, signal, alpha=None):
    ema_fast = wma(prices, fast, ema_filter(fast, alpha))
    ema_slow = wma(prices, slow, ema_filter(slow, alpha))
    min_len = min(len(ema_fast), len(ema_slow))
    macd_line = ema_fast[-min_len:] - ema_slow[-min_len:]
    signal_line = wma(macd_line, signal, ema_filter(signal, alpha))
    macd_line = macd_line[-len(signal_line):]
    return macd_line, signal_line
