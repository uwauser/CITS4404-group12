
initial_cash = 1000
fee = 0.03
timeframes = [1, 6, 12, 24, 72, 168]

verbose = True
log_convergence = True
random_seed = 42

bounds = [
    (0.0, 3.0),  # w1
    (0.0, 3.0),  # w2
    (0.0, 3.0),  # w3
    (0.0, 3.0),  # w4
    (3, 30),     # d1 (LMA)
    (3, 30),     # d2 (SMA)
    (3, 30),     # d3 (EMA)
    (0.05, 0.5), # alpha
    (3, 30),     # d4 (MACD fast)
    (10, 60),    # d5 (MACD slow)
    (3, 30),     # d6 (MACD signal)
    (10, 60),    # d7 (LOW SMA)
    (0, len(timeframes) - 1)  # tf
]

# Artificial Bee Colony
abc_settings = {
    "NUM_BEES": 30,
    "MAX_ITER": 20,
    "LIMIT": 5
}

# Particle Swarm
pso_settings = {
    "NUM_PARTICLES": 30,
    "MAX_ITER": 20,
    "w": 0.5,
    "c1": 1.5,
    "c2": 1.5
}

# Grey Wolf
gwo_settings = {
    "NUM_WOLVES": 30,
    "MAX_ITER": 20
}

# Whale
woa_settings = {
    "NUM_WHALES": 30,
    "MAX_ITER": 20
}

# Firefly
fa_settings = {
    "NUM_FIREFLIES": 30,
    "MAX_ITER": 20,
    "gamma": 1.0,
    "beta0": 1.0,
    "alpha": 0.2
}

# Cuckoo Search
cs_settings = {
    "NUM_NESTS": 30,
    "MAX_ITER": 20,
    "pa": 0.25
}

# Simulated Annealing
sa_settings = {
    "MAX_EVALS": 600,
    "initial_temp": 100.0,
    "cooling_rate": 0.95,
    "perturb_scale": 0.1
}