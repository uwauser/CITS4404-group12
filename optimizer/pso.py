import random
import numpy as np
from utils import quality, downsample
from config import bounds, timeframes, settings

def particle_swarm(price_series, log=None, setting=None):
    """
    Implements the Particle Swarm Optimization (PSO) algorithm for optimization.

    Args:
        price_series (list or np.ndarray): The input price series data.
        log (list, optional): A list to store the best fitness value at each generation. Defaults to None.
        setting (dict, optional): A dictionary containing PSO-specific settings. Defaults to the first setting in `settings["PSO"]`.

    Returns:
        tuple: A tuple containing:
            - gbest (np.ndarray): The best solution found by the algorithm.
            - gbest_fitness (float): The fitness value of the best solution.
            - best_timeframe (int): The timeframe corresponding to the best solution.
    """
    if setting is None:
        setting = settings["PSO"][0]  # Use default settings if none are provided.

    # Extract algorithm parameters from the settings.
    NUM_PARTICLES = setting["NUM_PARTICLES"]  # Number of particles in the swarm.
    MAX_ITER = setting["MAX_ITER"]  # Maximum number of generations.
    w = setting["w"]  # Inertia weight.
    c1 = setting["c1"]  # Cognitive coefficient.
    c2 = setting["c2"]  # Social coefficient.

    # Dimensionality of the problem (based on bounds).
    dim = len(bounds)

    # Initialize particles with random positions within the bounds.
    particles = [np.array([random.uniform(low, high) for (low, high) in bounds]) for _ in range(NUM_PARTICLES)]
    velocities = [np.zeros(dim) for _ in range(NUM_PARTICLES)]  # Initialize velocities to zero.

    # Calculate the initial fitness of each particle.
    fitness = [quality(p[:-1], downsample(price_series, timeframes[int(p[-1])])) for p in particles]

    # Initialize personal best positions and fitness values.
    pbest = particles.copy()
    pbest_fitness = fitness.copy()

    # Identify the global best particle.
    gbest_idx = np.argmax(fitness)
    gbest = particles[gbest_idx].copy()
    gbest_fitness = fitness[gbest_idx]

    # Main loop: iterate through generations.
    for gen in range(MAX_ITER):
        for i in range(NUM_PARTICLES):
            # Generate random coefficients for velocity update.
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            # Update velocity based on inertia, cognitive, and social components.
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (pbest[i] - particles[i]) +
                             c2 * r2 * (gbest - particles[i]))
            # Update particle position.
            particles[i] += velocities[i]

            # Ensure the particle stays within bounds.
            for d in range(dim - 1):
                particles[i][d] = max(min(particles[i][d], bounds[d][1]), bounds[d][0])
            particles[i][-1] = int(round(max(min(particles[i][-1], bounds[-1][1]), bounds[-1][0])))

            # Recalculate fitness for the updated particle.
            tf = timeframes[int(particles[i][-1])]
            fit = quality(particles[i][:-1], downsample(price_series, tf))

            # Update personal best if the new fitness is better.
            if fit > pbest_fitness[i]:
                pbest[i], pbest_fitness[i] = particles[i].copy(), fit

            # Update global best if the new fitness is better.
            if fit > gbest_fitness:
                gbest, gbest_fitness = particles[i].copy(), fit

        # Log the best fitness value if a log is provided.
        if log is not None:
            log.append(gbest_fitness)

        # Print the progress of the algorithm.
        print(f"PSO Gen {gen+1:02d} | Best Profit: ${gbest_fitness:.2f} | TF: {timeframes[int(gbest[-1])]}h")

    # Return the best solution found.
    return gbest, gbest_fitness, timeframes[int(gbest[-1])]