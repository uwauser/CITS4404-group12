import random
import numpy as np
from implementation.utils import quality, downsample
from implementation.config import bounds, timeframes, sa_settings

MAX_EVALS = sa_settings["MAX_EVALS"]
initial_temp = sa_settings["initial_temp"]
cooling_rate = sa_settings["cooling_rate"]
perturb_scale = sa_settings["perturb_scale"]

def simulated_annealing(price_series):
    eval_count = 0
    dim = len(bounds)

    def random_solution():
        sol = [random.uniform(low, high) for (low, high) in bounds[:-1]]
        sol.append(random.randint(0, len(timeframes) - 1))
        return np.array(sol)

    def perturb(sol):
        new_sol = sol.copy()
        idx = random.randint(0, dim - 2)
        low, high = bounds[idx]
        scale = (high - low) * perturb_scale
        new_sol[idx] += np.random.normal(0, scale)
        new_sol[idx] = max(min(new_sol[idx], high), low)
        new_sol[-1] = int(round(max(min(sol[-1], bounds[-1][1]), bounds[-1][0])))
        return new_sol

    curr_sol = random_solution()
    curr_fit = quality(curr_sol[:-1], downsample(price_series, timeframes[int(curr_sol[-1])]))
    best_sol, best_fit = curr_sol.copy(), curr_fit
    temp = initial_temp

    while eval_count < MAX_EVALS:
        new_sol = perturb(curr_sol)
        new_fit = quality(new_sol[:-1], downsample(price_series, timeframes[int(new_sol[-1])]))
        eval_count += 1

        if new_fit > curr_fit or random.random() < np.exp((new_fit - curr_fit) / temp):
            curr_sol, curr_fit = new_sol, new_fit

            if new_fit > best_fit:
                best_sol, best_fit = new_sol.copy(), new_fit

        temp *= cooling_rate

    print(f"SA Done | Best Profit: ${best_fit:.2f} | TF: {timeframes[int(best_sol[-1])]}h")
    return best_sol, best_fit, timeframes[int(best_sol[-1])]
