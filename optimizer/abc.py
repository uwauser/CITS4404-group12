import random
import numpy as np
from utils import quality, downsample
from config import bounds, timeframes, settings


def init_solution():
    """
    Initializes a random solution within the defined bounds.

    Returns:
        list: A solution where each element is a random value within the corresponding bounds.
              The last element is an integer index for the timeframes.
    """
    sol = [random.uniform(low, high) for low, high in bounds[:-1]]
    sol.append(random.randint(0, len(timeframes) - 1))
    return sol


def clamp(solution):
    """
    Clamps a solution to ensure all values are within their respective bounds.

    Args:
        solution (list): A solution to be clamped.

    Returns:
        list: A clamped solution where each value is within its respective bounds.
    """
    clamped = [max(min(val, high), low) for val, (low, high) in zip(solution[:-1], bounds[:-1])]
    clamped.append(int(round(max(min(solution[-1], bounds[-1][1]), bounds[-1][0]))))
    return clamped


def artificial_bee_colony(price_series, log=None, setting=None):
    """
    Implements the Artificial Bee Colony (ABC) optimization algorithm.

    Args:
        price_series (list): The input data series for optimization.
        log (list, optional): A list to store the best fitness values for each generation.
        setting (dict, optional): Configuration settings for the ABC algorithm. If None, defaults are used.

    Returns:
        tuple: A tuple containing:
            - best_solution (list): The best solution found.
            - best_fitness (float): The fitness value of the best solution.
            - best_timeframe (int): The timeframe index of the best solution.
    """
    if setting is None:
        setting = settings["ABC"][0]

    # Extract settings
    NUM_BEES = setting["NUM_BEES"]
    NUM_EMPLOYED = NUM_ONLOOKER = NUM_BEES // 2
    MAX_ITER = setting["MAX_ITER"]
    LIMIT = setting["LIMIT"]

    # Initialize population and fitness
    population = [init_solution() for _ in range(NUM_BEES)]
    fitness = [quality(sol[:-1], downsample(price_series, timeframes[int(sol[-1])])) for sol in population]
    trial = [0] * NUM_BEES

    # Track the best solution
    best_idx = np.argmax(fitness)
    best_solution = population[best_idx]
    best_fitness = fitness[best_idx]

    # Main optimization loop
    for gen in range(MAX_ITER):
        # Employed bee phase
        for i in range(NUM_EMPLOYED):
            k = random.choice([j for j in range(NUM_BEES) if j != i])  # Select a random neighbor
            phi = random.uniform(-1, 1)  # Random perturbation factor
            new_sol = population[i][:]
            dim = random.randint(0, len(bounds) - 2)  # Random dimension to modify
            new_sol[dim] += phi * (population[i][dim] - population[k][dim])
            new_sol = clamp(new_sol)  # Clamp the solution to bounds
            tf = timeframes[int(new_sol[-1])]
            new_fit = quality(new_sol[:-1], downsample(price_series, tf))
            if new_fit > fitness[i]:  # Update if the new solution is better
                population[i], fitness[i], trial[i] = new_sol, new_fit, 0
            else:
                trial[i] += 1

        # Onlooker bee phase
        max_fit = max(fitness)
        probs = [f / max_fit for f in fitness]  # Calculate selection probabilities
        for _ in range(NUM_ONLOOKER):
            i = random.choices(range(NUM_BEES), weights=probs)[0]  # Select a bee based on probabilities
            k = random.choice([j for j in range(NUM_BEES) if j != i])  # Select a random neighbor
            phi = random.uniform(-1, 1)  # Random perturbation factor
            new_sol = population[i][:]
            dim = random.randint(0, len(bounds) - 2)  # Random dimension to modify
            new_sol[dim] += phi * (population[i][dim] - population[k][dim])
            new_sol = clamp(new_sol)  # Clamp the solution to bounds
            tf = timeframes[int(new_sol[-1])]
            new_fit = quality(new_sol[:-1], downsample(price_series, tf))
            if new_fit > fitness[i]:  # Update if the new solution is better
                population[i], fitness[i], trial[i] = new_sol, new_fit, 0
            else:
                trial[i] += 1

        # Scout bee phase
        for i in range(NUM_BEES):
            if trial[i] >= LIMIT:  # Replace solution if it exceeds the trial limit
                population[i] = init_solution()
                tf = timeframes[int(population[i][-1])]
                fitness[i] = quality(population[i][:-1], downsample(price_series, tf))
                trial[i] = 0

        # Update the best solution
        best_idx = np.argmax(fitness)
        if fitness[best_idx] > best_fitness:
            best_solution = population[best_idx]
            best_fitness = fitness[best_idx]

        # Log the best fitness value
        if log is not None:
            log.append(best_fitness)

        # Print progress
        print(f"Gen {gen + 1:02d} | Best Profit: ${best_fitness:.2f} | TF: {timeframes[int(best_solution[-1])]}h")

    return best_solution, best_fitness, timeframes[int(best_solution[-1])]