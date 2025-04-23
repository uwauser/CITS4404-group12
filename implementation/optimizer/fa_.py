import random
import numpy as np
from implementation.utils import quality, downsample
from implementation.config import bounds, timeframes, fa_settings

NUM_FIREFLIES = fa_settings["NUM_FIREFLIES"]
MAX_ITER = fa_settings["MAX_ITER"]
gamma = fa_settings["gamma"]
beta0 = fa_settings["beta0"]
alpha = fa_settings["alpha"]

def firefly(price_series):
    dim = len(bounds)
    fireflies = [np.array([random.uniform(low, high) for (low, high) in bounds]) for _ in range(NUM_FIREFLIES)]
    fitness = [quality(f[:-1], downsample(price_series, timeframes[int(f[-1])])) for f in fireflies]

    for gen in range(MAX_ITER):
        for i in range(NUM_FIREFLIES):
            for j in range(NUM_FIREFLIES):
                if fitness[j] > fitness[i]:
                    r = np.linalg.norm(fireflies[i] - fireflies[j])
                    beta = beta0 * np.exp(-gamma * r ** 2)
                    rand = alpha * (np.random.rand(dim) - 0.5)
                    fireflies[i] = fireflies[i] + beta * (fireflies[j] - fireflies[i]) + rand

                    # Clamp
                    for d in range(dim - 1):
                        fireflies[i][d] = max(min(fireflies[i][d], bounds[d][1]), bounds[d][0])
                    fireflies[i][-1] = int(round(max(min(fireflies[i][-1], bounds[-1][1]), bounds[-1][0])))

                    tf = timeframes[int(fireflies[i][-1])]
                    fitness[i] = quality(fireflies[i][:-1], downsample(price_series, tf))

        best_idx = np.argmax(fitness)
        print(f"FA Gen {gen+1:02d} | Best Profit: ${fitness[best_idx]:.2f} | TF: {timeframes[int(fireflies[best_idx][-1])]}h")

    best_idx = np.argmax(fitness)
    return fireflies[best_idx], fitness[best_idx], timeframes[int(fireflies[best_idx][-1])]