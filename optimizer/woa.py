import random
import numpy as np
from utils import quality, downsample
from config import bounds, timeframes, settings

def whale(price_series, log=None, setting=None):
    """
    Implements the Whale Optimization Algorithm (WOA) for optimization.

    Args:
        price_series (list or np.ndarray): The input price series data.
        log (list, optional): A list to store the best fitness value at each generation. Defaults to None.
        setting (dict, optional): A dictionary containing WOA-specific settings. Defaults to the first setting in `settings["WOA"]`.

    Returns:
        tuple: A tuple containing:
            - best_whale (np.ndarray): The best solution found by the algorithm.
            - best_fitness (float): The fitness value of the best solution.
            - best_timeframe (int): The timeframe corresponding to the best solution.
    """
    if setting is None:
        setting = settings["WOA"][0]  # Use default settings if none are provided.

    # Extract algorithm parameters from the settings.
    NUM_WHALES = setting["NUM_WHALES"]  # Number of whales in the population.
    MAX_ITER = setting["MAX_ITER"]  # Maximum number of generations.
    dim = len(bounds)  # Dimensionality of the problem (based on bounds).

    # Initialize whales with random positions within the bounds.
    whales = [np.array([random.uniform(low, high) for (low, high) in bounds]) for _ in range(NUM_WHALES)]

    # Calculate the initial fitness of each whale.
    fitness = [quality(w[:-1], downsample(price_series, timeframes[int(w[-1])])) for w in whales]

    # Identify the best whale in the initial population.
    best_idx = np.argmax(fitness)
    best_whale = whales[best_idx].copy()
    best_fitness = fitness[best_idx]

    # Main loop: iterate through generations.
    for gen in range(MAX_ITER):
        # Linearly decrease the parameter 'a' over iterations.
        a = 2 - gen * (2 / MAX_ITER)

        for i in range(NUM_WHALES):
            r = random.random()  # Random value in [0, 1].
            A = 2 * a * r - a  # Calculate the coefficient A.
            C = 2 * random.random()  # Calculate the coefficient C.
            p = random.random()  # Probability to choose exploitation or exploration.

            if p < 0.5:
                if abs(A) >= 1:
                    # Exploration phase: move towards a random whale.
                    rand_idx = random.randint(0, NUM_WHALES - 1)
                    X_rand = whales[rand_idx]
                    D = abs(C * X_rand - whales[i])
                    new_pos = X_rand - A * D
                else:
                    # Exploitation phase: move towards the best whale.
                    D = abs(C * best_whale - whales[i])
                    new_pos = best_whale - A * D
            else:
                # Spiral updating position (exploitation phase).
                D = abs(best_whale - whales[i])
                b = 1  # Constant for logarithmic spiral.
                l = random.uniform(-1, 1)  # Random value in [-1, 1].
                new_pos = D * np.exp(b * l) * np.cos(2 * np.pi * l) + best_whale

            # Ensure the whale stays within bounds.
            for d in range(dim - 1):
                new_pos[d] = max(min(new_pos[d], bounds[d][1]), bounds[d][0])
            new_pos[-1] = int(round(max(min(new_pos[-1], bounds[-1][1]), bounds[-1][0])))

            # Recalculate fitness for the updated whale.
            tf = timeframes[int(new_pos[-1])]
            new_fit = quality(new_pos[:-1], downsample(price_series, tf))

            # Update the whale's position and fitness if the new position is better.
            if new_fit > fitness[i]:
                whales[i] = new_pos
                fitness[i] = new_fit

                # Update the best whale if the new fitness is better.
                if new_fit > best_fitness:
                    best_whale = new_pos
                    best_fitness = new_fit

        # Log the best fitness value if a log is provided.
        if log is not None:
            log.append(best_fitness)

        # Print the progress of the algorithm.
        print(f"WOA Gen {gen+1:02d} | Best Profit: ${best_fitness:.2f} | TF: {timeframes[int(best_whale[-1])]}h")

    # Return the best solution found.
    return best_whale, best_fitness, timeframes[int(best_whale[-1])]