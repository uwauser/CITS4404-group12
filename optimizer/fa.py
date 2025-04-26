import random
import numpy as np
from utils import quality, downsample
from config import bounds, timeframes, settings

def firefly(price_series, log=None, setting=None):
    """
    Implements the Firefly Algorithm (FA) for optimization.

    Args:
        price_series (list or np.ndarray): The input price series data.
        log (list, optional): A list to store the best fitness value at each generation. Defaults to None.
        setting (dict, optional): A dictionary containing FA-specific settings. Defaults to the first setting in `settings["FA"]`.

    Returns:
        tuple: A tuple containing:
            - best_firefly (np.ndarray): The best solution found by the algorithm.
            - best_fitness (float): The fitness value of the best solution.
            - best_timeframe (int): The timeframe corresponding to the best solution.
    """
    if setting is None:
        setting = settings["FA"][0]  # Use default settings if none are provided.

    # Extract algorithm parameters from the settings.
    NUM_FIREFLIES = setting["NUM_FIREFLIES"]  # Number of fireflies in the population.
    MAX_ITER = setting["MAX_ITER"]  # Maximum number of generations.
    gamma = setting["gamma"]  # Light absorption coefficient.
    beta0 = setting["beta0"]  # Base attractiveness.
    alpha = setting["alpha"]  # Randomness scaling factor.

    # Dimensionality of the problem (based on bounds).
    dim = len(bounds)

    # Initialize fireflies with random positions within the bounds.
    fireflies = [np.array([random.uniform(low, high) for (low, high) in bounds]) for _ in range(NUM_FIREFLIES)]

    # Calculate the initial fitness of each firefly.
    fitness = [quality(f[:-1], downsample(price_series, timeframes[int(f[-1])])) for f in fireflies]

    # Main loop: iterate through generations.
    for gen in range(MAX_ITER):
        # Update each firefly based on its neighbors.
        for i in range(NUM_FIREFLIES):
            for j in range(NUM_FIREFLIES):
                if fitness[j] > fitness[i]:  # Move firefly i towards firefly j if j is brighter.
                    r = np.linalg.norm(fireflies[i] - fireflies[j])  # Distance between fireflies.
                    beta = beta0 * np.exp(-gamma * r ** 2)  # Calculate attractiveness.
                    rand = alpha * (np.random.rand(dim) - 0.5)  # Random perturbation.
                    fireflies[i] = fireflies[i] + beta * (fireflies[j] - fireflies[i]) + rand

                    # Ensure the firefly stays within bounds.
                    for d in range(dim - 1):
                        fireflies[i][d] = max(min(fireflies[i][d], bounds[d][1]), bounds[d][0])
                    fireflies[i][-1] = int(round(max(min(fireflies[i][-1], bounds[-1][1]), bounds[-1][0])))

                    # Recalculate fitness for the updated firefly.
                    tf = timeframes[int(fireflies[i][-1])]
                    fitness[i] = quality(fireflies[i][:-1], downsample(price_series, tf))

        # Identify the best firefly in the current generation.
        best_idx = np.argmax(fitness)

        # Log the best fitness value if a log is provided.
        if log is not None:
            log.append(fitness[best_idx])

        # Print the progress of the algorithm.
        print(f"FA Gen {gen+1:02d} | Best Profit: ${fitness[best_idx]:.2f} | TF: {timeframes[int(fireflies[best_idx][-1])]}h")

    # Return the best solution found.
    best_idx = np.argmax(fitness)
    return fireflies[best_idx], fitness[best_idx], timeframes[int(fireflies[best_idx][-1])]