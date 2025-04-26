import random
import numpy as np
from utils import quality, downsample
from config import bounds, timeframes, settings

def grey_wolf(price_series, log=None, setting=None):
    """
    Implements the Grey Wolf Optimizer (GWO) algorithm for optimization.

    Args:
        price_series (list or np.ndarray): The input price series data.
        log (list, optional): A list to store the best fitness value at each generation. Defaults to None.
        setting (dict, optional): A dictionary containing GWO-specific settings. Defaults to the first setting in `settings["GWO"]`.

    Returns:
        tuple: A tuple containing:
            - best_wolf (np.ndarray): The best solution found by the algorithm.
            - best_fitness (float): The fitness value of the best solution.
            - best_timeframe (int): The timeframe corresponding to the best solution.
    """
    if setting is None:
        setting = settings["GWO"][0]  # Use default settings if none are provided.

    # Extract algorithm parameters from the settings.
    NUM_WOLVES = setting["NUM_WOLVES"]  # Number of wolves in the population.
    MAX_ITER = setting["MAX_ITER"]  # Maximum number of generations.

    # Dimensionality of the problem (based on bounds).
    dim = len(bounds)

    # Initialize wolves with random positions within the bounds.
    wolves = [np.array([random.uniform(low, high) for (low, high) in bounds]) for _ in range(NUM_WOLVES)]

    # Calculate the initial fitness of each wolf.
    fitness = [quality(w[:-1], downsample(price_series, timeframes[int(w[-1])])) for w in wolves]

    # Main loop: iterate through generations.
    for gen in range(MAX_ITER):
        # Sort wolves by fitness in descending order.
        sorted_indices = np.argsort(fitness)[::-1]
        alpha, beta, delta = [wolves[i] for i in sorted_indices[:3]]  # Top three wolves.

        # Linearly decrease the parameter 'a' over iterations.
        a = 2 - gen * (2 / MAX_ITER)

        # Update the position of each wolf.
        for i in range(NUM_WOLVES):
            # Calculate distances to alpha, beta, and delta wolves.
            D_alpha = abs(2 * random.random() * alpha - wolves[i])
            D_beta = abs(2 * random.random() * beta - wolves[i])
            D_delta = abs(2 * random.random() * delta - wolves[i])

            # Calculate new positions based on alpha, beta, and delta.
            X1 = alpha - a * D_alpha
            X2 = beta - a * D_beta
            X3 = delta - a * D_delta

            # Average the positions to get the new position.
            new_pos = (X1 + X2 + X3) / 3

            # Ensure the wolf stays within bounds.
            for d in range(dim - 1):
                new_pos[d] = max(min(new_pos[d], bounds[d][1]), bounds[d][0])
            new_pos[-1] = int(round(max(min(new_pos[-1], bounds[-1][1]), bounds[-1][0])))

            # Recalculate fitness for the updated wolf.
            tf = timeframes[int(new_pos[-1])]
            new_fit = quality(new_pos[:-1], downsample(price_series, tf))

            # Update the wolf's position and fitness if the new position is better.
            if new_fit > fitness[i]:
                wolves[i] = new_pos
                fitness[i] = new_fit

        # Identify the best wolf in the current generation.
        best_idx = np.argmax(fitness)

        # Log the best fitness value if a log is provided.
        if log is not None:
            log.append(fitness[best_idx])

        # Print the progress of the algorithm.
        print(f"GWO Gen {gen+1:02d} | Best Profit: ${fitness[best_idx]:.2f} | TF: {timeframes[int(wolves[best_idx][-1])]}h")

    # Return the best solution found.
    best_idx = np.argmax(fitness)
    return wolves[best_idx], fitness[best_idx], timeframes[int(wolves[best_idx][-1])]