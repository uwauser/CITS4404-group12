import random
import numpy as np
from implementation.utils import quality, downsample
from implementation.config import bounds, timeframes, gwo_settings

NUM_WOLVES = gwo_settings["NUM_WOLVES"]
MAX_ITER = gwo_settings["MAX_ITER"]

def grey_wolf(price_series, log=None):
    dim = len(bounds)
    wolves = [np.array([random.uniform(low, high) for (low, high) in bounds]) for _ in range(NUM_WOLVES)]
    fitness = [quality(w[:-1], downsample(price_series, timeframes[int(w[-1])])) for w in wolves]

    for gen in range(MAX_ITER):
        sorted_indices = np.argsort(fitness)[::-1]
        alpha, beta, delta = [wolves[i] for i in sorted_indices[:3]]

        a = 2 - gen * (2 / MAX_ITER)

        for i in range(NUM_WOLVES):
            D_alpha = abs(2 * random.random() * alpha - wolves[i])
            D_beta = abs(2 * random.random() * beta - wolves[i])
            D_delta = abs(2 * random.random() * delta - wolves[i])

            X1 = alpha - a * D_alpha
            X2 = beta - a * D_beta
            X3 = delta - a * D_delta

            new_pos = (X1 + X2 + X3) / 3

            for d in range(dim - 1):
                new_pos[d] = max(min(new_pos[d], bounds[d][1]), bounds[d][0])
            new_pos[-1] = int(round(max(min(new_pos[-1], bounds[-1][1]), bounds[-1][0])))

            tf = timeframes[int(new_pos[-1])]
            new_fit = quality(new_pos[:-1], downsample(price_series, tf))

            if new_fit > fitness[i]:
                wolves[i] = new_pos
                fitness[i] = new_fit

        best_idx = np.argmax(fitness)

        if log is not None:
            log.append(fitness[best_idx])

        print(f"GWO Gen {gen+1:02d} | Best Profit: ${fitness[best_idx]:.2f} | TF: {timeframes[int(wolves[best_idx][-1])]}h")

    best_idx = np.argmax(fitness)
    return wolves[best_idx], fitness[best_idx], timeframes[int(wolves[best_idx][-1])]