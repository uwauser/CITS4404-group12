import random
import numpy as np
from implementation.utils import quality, downsample
from implementation.config import bounds, timeframes, abc_settings

NUM_BEES = abc_settings["NUM_BEES"]
NUM_EMPLOYED = NUM_ONLOOKER = NUM_BEES // 2
MAX_ITER = abc_settings["MAX_ITER"]
LIMIT = abc_settings["LIMIT"]

def init_solution():
    sol = [random.uniform(low, high) for low, high in bounds[:-1]]
    sol.append(random.randint(0, len(timeframes) - 1))
    return sol

def clamp(solution):
    clamped = [max(min(val, high), low) for val, (low, high) in zip(solution[:-1], bounds[:-1])]
    clamped.append(int(round(max(min(solution[-1], bounds[-1][1]), bounds[-1][0]))))
    return clamped

def artificial_bee_colony(price_series, log=None):
    population = [init_solution() for _ in range(NUM_BEES)]
    fitness = [quality(sol[:-1], downsample(price_series, timeframes[int(sol[-1])])) for sol in population]
    trial = [0] * NUM_BEES

    best_idx = np.argmax(fitness)
    best_solution = population[best_idx]
    best_fitness = fitness[best_idx]

    for gen in range(MAX_ITER):
        for i in range(NUM_EMPLOYED):
            k = random.choice([j for j in range(NUM_BEES) if j != i])
            phi = random.uniform(-1, 1)
            new_sol = population[i][:]
            dim = random.randint(0, len(bounds) - 2)
            new_sol[dim] += phi * (population[i][dim] - population[k][dim])
            new_sol = clamp(new_sol)
            tf = timeframes[int(new_sol[-1])]
            new_fit = quality(new_sol[:-1], downsample(price_series, tf))
            if new_fit > fitness[i]:
                population[i], fitness[i], trial[i] = new_sol, new_fit, 0
            else:
                trial[i] += 1

        max_fit = max(fitness)
        probs = [f / max_fit for f in fitness]
        for _ in range(NUM_ONLOOKER):
            i = random.choices(range(NUM_BEES), weights=probs)[0]
            k = random.choice([j for j in range(NUM_BEES) if j != i])
            phi = random.uniform(-1, 1)
            new_sol = population[i][:]
            dim = random.randint(0, len(bounds) - 2)
            new_sol[dim] += phi * (population[i][dim] - population[k][dim])
            new_sol = clamp(new_sol)
            tf = timeframes[int(new_sol[-1])]
            new_fit = quality(new_sol[:-1], downsample(price_series, tf))
            if new_fit > fitness[i]:
                population[i], fitness[i], trial[i] = new_sol, new_fit, 0
            else:
                trial[i] += 1

        for i in range(NUM_BEES):
            if trial[i] >= LIMIT:
                population[i] = init_solution()
                tf = timeframes[int(population[i][-1])]
                fitness[i] = quality(population[i][:-1], downsample(price_series, tf))
                trial[i] = 0

        best_idx = np.argmax(fitness)
        if fitness[best_idx] > best_fitness:
            best_solution = population[best_idx]
            best_fitness = fitness[best_idx]

        if log is not None:
            log.append(best_fitness)

        print(f"Gen {gen+1:02d} | Best Profit: ${best_fitness:.2f} | TF: {timeframes[int(best_solution[-1])]}h")

    return best_solution, best_fitness, timeframes[int(best_solution[-1])]