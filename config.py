
seed = 42
initial_cash = 1000
fee = 0.03
timeframes = [1, 6, 12, 24, 72, 168]

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

settings = {
    "ABC": [
        {"NUM_BEES": 10, "MAX_ITER": 60, "LIMIT": 10},
        {"NUM_BEES": 20, "MAX_ITER": 30, "LIMIT": 5},
        {"NUM_BEES": 30, "MAX_ITER": 20, "LIMIT": 5},
        {"NUM_BEES": 40, "MAX_ITER": 15, "LIMIT": 4},
        {"NUM_BEES": 60, "MAX_ITER": 10, "LIMIT": 3},
    ],
    "PSO": [
        {"NUM_PARTICLES": 10, "MAX_ITER": 60, "w": 0.4, "c1": 1.5, "c2": 1.5},
        {"NUM_PARTICLES": 20, "MAX_ITER": 30, "w": 0.6, "c1": 1.0, "c2": 2.0},
        {"NUM_PARTICLES": 30, "MAX_ITER": 20, "w": 0.5, "c1": 1.5, "c2": 1.5},
        {"NUM_PARTICLES": 40, "MAX_ITER": 15, "w": 0.3, "c1": 2.0, "c2": 2.0},
        {"NUM_PARTICLES": 60, "MAX_ITER": 10, "w": 0.8, "c1": 1.2, "c2": 1.2},
    ],
    "GWO": [
        {"NUM_WOLVES": 10, "MAX_ITER": 60},
        {"NUM_WOLVES": 20, "MAX_ITER": 30},
        {"NUM_WOLVES": 30, "MAX_ITER": 20},
        {"NUM_WOLVES": 40, "MAX_ITER": 15},
        {"NUM_WOLVES": 60, "MAX_ITER": 10},
    ],
    "WOA": [
        {"NUM_WHALES": 10, "MAX_ITER": 60},
        {"NUM_WHALES": 20, "MAX_ITER": 30},
        {"NUM_WHALES": 30, "MAX_ITER": 20},
        {"NUM_WHALES": 40, "MAX_ITER": 15},
        {"NUM_WHALES": 60, "MAX_ITER": 10},
    ],
    "FA": [
        {"NUM_FIREFLIES": 10, "MAX_ITER": 60, "gamma": 1.0, "beta0": 1.0, "alpha": 0.2},
        {"NUM_FIREFLIES": 20, "MAX_ITER": 30, "gamma": 0.5, "beta0": 1.5, "alpha": 0.3},
        {"NUM_FIREFLIES": 30, "MAX_ITER": 20, "gamma": 1.0, "beta0": 1.0, "alpha": 0.2},
        {"NUM_FIREFLIES": 40, "MAX_ITER": 15, "gamma": 0.9, "beta0": 0.9, "alpha": 0.1},
        {"NUM_FIREFLIES": 60, "MAX_ITER": 10, "gamma": 1.2, "beta0": 0.8, "alpha": 0.05},
    ],
    "CS": [
        {"NUM_NESTS": 10, "MAX_ITER": 60, "pa": 0.2},
        {"NUM_NESTS": 20, "MAX_ITER": 30, "pa": 0.25},
        {"NUM_NESTS": 30, "MAX_ITER": 20, "pa": 0.25},
        {"NUM_NESTS": 40, "MAX_ITER": 15, "pa": 0.3},
        {"NUM_NESTS": 60, "MAX_ITER": 10, "pa": 0.15},
    ],
    "SA": [
        {"MAX_EVALS": 600, "initial_temp": 100.0, "cooling_rate": 0.95, "perturb_scale": 0.1},
        {"MAX_EVALS": 600, "initial_temp": 200.0, "cooling_rate": 0.97, "perturb_scale": 0.05},
        {"MAX_EVALS": 600, "initial_temp": 50.0,  "cooling_rate": 0.9,  "perturb_scale": 0.2},
        {"MAX_EVALS": 600, "initial_temp": 150.0, "cooling_rate": 0.92, "perturb_scale": 0.15},
        {"MAX_EVALS": 600, "initial_temp": 100.0, "cooling_rate": 0.95, "perturb_scale": 0.3},
    ]
}