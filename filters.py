import numpy as np
from config import seed

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

def sma_filter(window):
    """
    Creates a Simple Moving Average (SMA) filter kernel.

    Args:
        window (int): The window size for the SMA filter.

    Returns:
        np.ndarray: The SMA filter kernel.
    """
    return np.ones(window) / window

def lma_filter(window):
    """
    Creates a Linear Moving Average (LMA) filter kernel.

    Args:
        window (int): The window size for the LMA filter.

    Returns:
        np.ndarray: The LMA filter kernel.
    """
    kernel = np.arange(window)
    weights = (2 / (window + 1)) * (1 - kernel / window)
    return weights / np.sum(weights)

def ema_filter(window, alpha):
    """
    Creates an Exponential Moving Average (EMA) filter kernel.

    Args:
        window (int): The window size for the EMA filter.
        alpha (float or None): The smoothing factor. If None, it defaults to 2 / (window + 1).

    Returns:
        np.ndarray: The EMA filter kernel.
    """
    if alpha is None:
        alpha = 2 / (window + 1)
    kernel = np.arange(window)
    weights = alpha * (1 - alpha) ** kernel
    return weights / np.sum(weights)

def triangle_filter(window):
    """
    Creates a triangular filter kernel.

    Args:
        window (int): The window size for the triangular filter.

    Returns:
        np.ndarray: The triangular filter kernel.
    """
    if window % 2 == 0:
        half = np.arange(1, window // 2 + 1)
        weights = np.concatenate([half, half[::-1]])
    else:
        half = np.arange(1, (window + 1) // 2 + 1)
        weights = np.concatenate([half, half[-2::-1]])
    return weights / np.sum(weights)

def gaussian_filter(window, sigma=None):
    """
    Creates a Gaussian filter kernel.

    Args:
        window (int): The window size for the Gaussian filter.
        sigma (float or None): The standard deviation of the Gaussian. Defaults to window / 6.

    Returns:
        np.ndarray: The Gaussian filter kernel.
    """
    if sigma is None:
        sigma = window / 6
    kernel = np.arange(window)
    center = (window - 1) / 2
    weights = np.exp(-0.5 * ((kernel - center) / sigma) ** 2)
    return weights / np.sum(weights)

def step_filter(window, amplitude=2.0):
    """
    Creates a step filter kernel with a specified amplitude.

    Args:
        window (int): The window size for the step filter.
        amplitude (float, optional): The amplitude of the filter. Defaults to 2.0.

    Returns:
        np.ndarray: The step filter kernel.
    """
    weights = np.ones(window)
    weights = weights / np.sum(weights)
    weights *= amplitude
    return weights

def random_filter(window):
    """
    Creates a random filter kernel.

    Args:
        window (int): The window size for the random filter.

    Returns:
        np.ndarray: The random filter kernel.
    """
    np.random.seed(seed)
    weights = np.random.rand(window)
    return weights / np.sum(weights)

def polynomial_filter(window, degree=1):
    """
    Creates a polynomial filter kernel.

    Args:
        window (int): The window size for the polynomial filter.
        degree (int, optional): The degree of the polynomial. Defaults to 1.

    Returns:
        np.ndarray: The polynomial filter kernel.
    """
    kernel = np.arange(window)
    weights = kernel ** degree
    return weights / np.sum(weights)

def polynomial_filter_centered(window, degree=2):
    """
    Creates a centered polynomial filter kernel.

    Args:
        window (int): The window size for the polynomial filter.
        degree (int, optional): The degree of the polynomial. Defaults to 2.

    Returns:
        np.ndarray: The centered polynomial filter kernel.
    """
    kernel = np.arange(window)
    center = (window - 1) / 2
    weights = np.abs(kernel - center) ** degree
    return weights / np.sum(weights)

def wma(prices, window, kernel):
    """
    Computes the Weighted Moving Average (WMA) of a price series.

    Args:
        prices (np.ndarray): The input price series.
        window (int): The window size for the WMA.
        kernel (np.ndarray): The filter kernel to apply.

    Returns:
        np.ndarray: The WMA of the price series.
    """
    return np.convolve(pad(prices, window), kernel, mode='valid')

def macd(prices, fast, slow, signal, alpha=None):
    """
    Computes the Moving Average Convergence Divergence (MACD) indicator.

    Args:
        prices (np.ndarray): The input price series.
        fast (int): The window size for the fast EMA.
        slow (int): The window size for the slow EMA.
        signal (int): The window size for the signal line.
        alpha (float or None, optional): The smoothing factor for the EMA. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - macd_line (np.ndarray): The MACD line.
            - signal_line (np.ndarray): The signal line.
    """
    ema_fast = wma(prices, fast, ema_filter(fast, alpha))
    ema_slow = wma(prices, slow, ema_filter(slow, alpha))
    min_len = min(len(ema_fast), len(ema_slow))
    macd_line = ema_fast[-min_len:] - ema_slow[-min_len:]
    signal_line = wma(macd_line, signal, ema_filter(signal, alpha))
    macd_line = macd_line[-len(signal_line):]
    return macd_line, signal_line