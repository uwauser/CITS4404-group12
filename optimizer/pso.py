import random
import numpy as np
from utils import quality, downsample
from config import bounds, timeframes, settings

def particle_swarm(price_series, log=None, setting=None):
    if setting is None:
        setting = settings["PSO"][0]
    NUM_PARTICLES = setting["NUM_PARTICLES"]
    MAX_ITER = setting["MAX_ITER"]
    w = setting["w"]
    c1 = setting["c1"]
    c2 = setting["c2"]
    dim = len(bounds)
    particles = [np.array([random.uniform(low, high) for (low, high) in bounds]) for _ in range(NUM_PARTICLES)]
    velocities = [np.zeros(dim) for _ in range(NUM_PARTICLES)]

    fitness = [quality(p[:-1], downsample(price_series, timeframes[int(p[-1])])) for p in particles]
    pbest = particles.copy()
    pbest_fitness = fitness.copy()

    gbest_idx = np.argmax(fitness)
    gbest = particles[gbest_idx].copy()
    gbest_fitness = fitness[gbest_idx]

    for gen in range(MAX_ITER):
        for i in range(NUM_PARTICLES):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (pbest[i] - particles[i]) +
                             c2 * r2 * (gbest - particles[i]))
            particles[i] += velocities[i]

            # Clamp
            for d in range(dim - 1):
                particles[i][d] = max(min(particles[i][d], bounds[d][1]), bounds[d][0])
            particles[i][-1] = int(round(max(min(particles[i][-1], bounds[-1][1]), bounds[-1][0])))

            tf = timeframes[int(particles[i][-1])]
            fit = quality(particles[i][:-1], downsample(price_series, tf))

            if fit > pbest_fitness[i]:
                pbest[i], pbest_fitness[i] = particles[i].copy(), fit

            if fit > gbest_fitness:
                gbest, gbest_fitness = particles[i].copy(), fit

        if log is not None:
            log.append(gbest_fitness)

        print(f"PSO Gen {gen+1:02d} | Best Profit: ${gbest_fitness:.2f} | TF: {timeframes[int(gbest[-1])]}h")

    return gbest, gbest_fitness, timeframes[int(gbest[-1])]