"""
Genetic Algorithm for Feature Selection using DEAP.
Selects optimal subset of features (method names) using
walk-forward validation and Random Forest regression.
Falls back gracefully if numpy/sklearn binary incompatible.
"""

import random
import sys

# Try to import numpy and sklearn — fallback gracefully if binary incompatible
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None
    _HAS_NUMPY = False

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    from deap import base, creator, tools, algorithms
    _HAS_SKBUILT = True
except (ImportError, ValueError):
    _HAS_SKBUILT = False
    RandomForestRegressor = None
    mean_squared_error = None
    base = None
    creator = None
    tools = None
    algorithms = None

from collections import deque


class FeatureSelector:
    """Genetic Algorithm-based feature selector using DEAP."""

    def __init__(self, X, y, method_names):
        if np is None or not _HAS_NUMPY:
            raise RuntimeError("numpy not available for FeatureSelector")
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.method_names = list(method_names)
        self.selected_features = None
        self.weights = None

    def evaluate_individual(self, individual):
        selected_idx = [i for i, bit in enumerate(individual) if bit == 1]
        if len(selected_idx) == 0:
            return (1e6,)

        X_sub = self.X[:, selected_idx]
        n = X_sub.shape[0]
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        X_train, y_train = X_sub[:train_end], self.y[:train_end]
        X_val, y_val = X_sub[train_end:val_end], self.y[train_end:val_end]

        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        complexity_penalty = len(selected_idx) * 0.0001
        return (mse + complexity_penalty,)

    def run_ga(self, n_generations=20, population_size=30):
        if not _HAS_SKBUILT:
            print("[FeatureSelector] sklearn/DEAP not available, skipping GA", file=sys.stderr)
            return []

        n_features = self.X.shape[1]

        for t in ("FitnessMin", "Individual"):
            if hasattr(creator, t):
                delattr(creator, t)

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.evaluate_individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=population_size)
        algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=n_generations, verbose=True)

        best_individual = tools.selBest(pop, k=1)[0]
        best_idx = [i for i, bit in enumerate(best_individual) if bit == 1]
        self.selected_features = [self.method_names[i] for i in best_idx]
        self.weights = best_individual
        return self.selected_features

    def get_selected_features(self):
        return self.selected_features


def weekly_feature_optimization():
    """Placeholder for weekly scheduled feature optimisation."""
    print("[weekly_feature_optimization] Checking historical data...", file=sys.stderr)
    if not _HAS_SKBUILT:
        print("[weekly_feature_optimization] sklearn/DEAP not available. Skipping.", file=sys.stderr)
        return []
    print("[weekly_feature_optimization] No historical data available. Skipping.", file=sys.stderr)
    return []
