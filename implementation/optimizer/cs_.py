import random
import numpy as np
from implementation.utils import quality, downsample
from implementation.config import bounds, timeframes, cs_settings

NUM_NESTS = cs_settings["NUM_NESTS"]
MAX_ITER = cs_settings["MAX_ITER"]
pa = cs_settings["pa"]

def levy_flight(Lambda):
    u = np.random.normal(0, 1)
    v = np.random.normal(0, 1)
    step = u / abs(v) ** (1 / Lambda)
    return step

def cuckoo_search(price_series):
    dim = len(bounds)
    nests = [np.array([random.uniform(low, high) for (low, high) in bounds]) for _ in range(NUM_NESTS)]
    fitness = [quality(n[:-1], downsample(price_series, timeframes[int(n[-1])])) for n in nests]

    best_idx = np.argmax(fitness)
    best_nest = nests[best_idx].copy()
    best_fitness = fitness[best_idx]

    for gen in range(MAX_ITER):
        for i in range(NUM_NESTS):
            step_size = levy_flight(1.5)
            new_nest = nests[i] + step_size * (nests[i] - best_nest)
            for d in range(dim - 1):
                new_nest[d] = max(min(new_nest[d], bounds[d][1]), bounds[d][0])
            new_nest[-1] = int(round(max(min(new_nest[-1], bounds[-1][1]), bounds[-1][0])))

            tf = timeframes[int(new_nest[-1])]
            new_fit = quality(new_nest[:-1], downsample(price_series, tf))

            if new_fit > fitness[i]:
                nests[i] = new_nest
                fitness[i] = new_fit

                if new_fit > best_fitness:
                    best_nest = new_nest
                    best_fitness = new_fit

        # Abandon some nests
        for i in range(NUM_NESTS):
            if random.random() < pa:
                nests[i] = np.array([random.uniform(low, high) for (low, high) in bounds])
                tf = timeframes[int(nests[i][-1])]
                fitness[i] = quality(nests[i][:-1], downsample(price_series, tf))

        print(f"CS Gen {gen+1:02d} | Best Profit: ${best_fitness:.2f} | TF: {timeframes[int(best_nest[-1])]}h")

    return best_nest, best_fitness, timeframes[int(best_nest[-1])]