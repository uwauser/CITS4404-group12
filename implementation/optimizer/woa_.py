import random
import numpy as np
from implementation.utils import quality, downsample
from implementation.config import bounds, timeframes, woa_settings

NUM_WHALES = woa_settings["NUM_WHALES"]
MAX_ITER = woa_settings["MAX_ITER"]

def whale(price_series, log=None):
    dim = len(bounds)
    whales = [np.array([random.uniform(low, high) for (low, high) in bounds]) for _ in range(NUM_WHALES)]
    fitness = [quality(w[:-1], downsample(price_series, timeframes[int(w[-1])])) for w in whales]

    best_idx = np.argmax(fitness)
    best_whale = whales[best_idx].copy()
    best_fitness = fitness[best_idx]

    for gen in range(MAX_ITER):
        a = 2 - gen * (2 / MAX_ITER)

        for i in range(NUM_WHALES):
            r = random.random()
            A = 2 * a * r - a
            C = 2 * random.random()
            p = random.random()

            if p < 0.5:
                if abs(A) >= 1:
                    rand_idx = random.randint(0, NUM_WHALES - 1)
                    X_rand = whales[rand_idx]
                    D = abs(C * X_rand - whales[i])
                    new_pos = X_rand - A * D
                else:
                    D = abs(C * best_whale - whales[i])
                    new_pos = best_whale - A * D
            else:
                D = abs(best_whale - whales[i])
                b = 1
                l = random.uniform(-1, 1)
                new_pos = D * np.exp(b * l) * np.cos(2 * np.pi * l) + best_whale

            for d in range(dim - 1):
                new_pos[d] = max(min(new_pos[d], bounds[d][1]), bounds[d][0])
            new_pos[-1] = int(round(max(min(new_pos[-1], bounds[-1][1]), bounds[-1][0])))

            tf = timeframes[int(new_pos[-1])]
            new_fit = quality(new_pos[:-1], downsample(price_series, tf))

            if new_fit > fitness[i]:
                whales[i] = new_pos
                fitness[i] = new_fit

                if new_fit > best_fitness:
                    best_whale = new_pos
                    best_fitness = new_fit

        if log is not None:
            log.append(best_fitness)

        print(f"WOA Gen {gen+1:02d} | Best Profit: ${best_fitness:.2f} | TF: {timeframes[int(best_whale[-1])]}h")

    return best_whale, best_fitness, timeframes[int(best_whale[-1])]