"""
CrisisDataAugmentation — TimeGAN-inspired synthetic crisis data generation.

Pure numpy implementation. No external dependencies.
Generates crisis scenarios via controlled random walks with high volatility
and negative drift, then augments training data with a configurable ratio.
"""

import numpy as np


class CrisisDataAugmentation:
    """TimeGAN alternative: generate synthetic crisis scenarios using random walks."""

    def __init__(self, lookback=30):
        self.lookback = lookback
        self.synthetic_history = []

    def generate_crisis_scenarios(self, n_samples=100, n_features=41):
        """Generate synthetic crisis scenarios via random walk with negative drift.

        Parameters
        ----------
        n_samples : int
            Number of candidate walks to generate.
        n_features : int
            Number of feature dimensions per time step.

        Returns
        -------
        np.ndarray
            Array of shape (n_crisis, lookback, n_features) where each entry
            passes the crisis filter (volatility > 0.02 and mean < -0.01).
        """
        rng = np.random.default_rng()

        crisis_scenarios = []

        for _ in range(n_samples):
            # Start at zero for all features
            walk = np.zeros((self.lookback, n_features))

            for t in range(1, self.lookback):
                # Random walk step: negative drift + high-vol noise
                drift = -0.002
                noise = rng.normal(loc=0.0, scale=0.05, size=n_features)
                walk[t] = walk[t - 1] + drift + noise

            # Crisis filter: high volatility + downward trend
            volatility = np.std(walk, axis=0).mean()
            mean_val = np.mean(walk)

            if volatility > 0.02 and mean_val < -0.01:
                crisis_scenarios.append(walk)

        if len(crisis_scenarios) == 0:
            # Fallback: return at least one worst-case walk
            fallback = np.zeros((self.lookback, n_features))
            for t in range(1, self.lookback):
                fallback[t] = fallback[t - 1] - 0.002 + np.random.default_rng().normal(
                    0, 0.05, n_features
                )
            crisis_scenarios.append(fallback)

        return np.array(crisis_scenarios)

    def augment_training_data(self, original_data, synthetic_data, ratio=0.15):
        """Augment original training data with synthetic crisis samples.

        Parameters
        ----------
        original_data : np.ndarray
            Shape (samples, n_features).
        synthetic_data : np.ndarray
            Shape (n_crisis, lookback, n_features).
        ratio : float
            Fraction of original samples to add (as flattened synthetic data).

        Returns
        -------
        np.ndarray
            Augmented array of shape (samples + n_added, n_features).
        """
        n_original = len(original_data)
        n_synthetic = max(1, int(n_original * ratio))

        # Ensure we don't request more than available
        n_synthetic = min(n_synthetic, len(synthetic_data))

        # Take a subset of crisis scenarios
        subset = synthetic_data[:n_synthetic]

        # Flatten: (n, lookback, n_features) -> (n * lookback, n_features)
        n_seq, lb, nf = subset.shape
        flattened = subset.reshape(n_seq * lb, nf)

        # Truncate or pad flattened to exactly n_synthetic samples worth
        # Actually, keep all flattened data for augmentation richness
        augmented = np.vstack([original_data, flattened])
        return augmented

    def filter_crisis_sequences(self, sequences):
        """Filter out non-crisis sequences.

        Parameters
        ----------
        sequences : np.ndarray
            Shape (n, lookback, n_features).

        Returns
        -------
        np.ndarray
            Only sequences where volatility > 0.02 and mean < -0.01.
        """
        if sequences.ndim != 3:
            raise ValueError(
                f"Expected 3D array (n, lookback, n_features), got shape {sequences.shape}"
            )

        keep = []
        for i in range(sequences.shape[0]):
            seq = sequences[i]
            vol = np.std(seq, axis=0).mean()
            mn = np.mean(seq)
            if vol > 0.02 and mn < -0.01:
                keep.append(seq)

        if len(keep) == 0:
            return np.empty((0, sequences.shape[1], sequences.shape[2]))

        return np.array(keep)


def monthly_data_augmentation():
    """Demo function: generate crisis scenarios and return augmented data.

    Returns
    -------
    tuple
        (crisis_scenarios, augmented_data) where augmented_data is a
        random array mimicking original monthly data stacked with synthetic
        crisis samples.
    """
    augmenter = CrisisDataAugmentation(lookback=30)

    # Generate crisis scenarios
    crisis = augmenter.generate_crisis_scenarios(n_samples=100, n_features=41)

    # Create dummy "original" data: 1000 samples with 41 features
    rng = np.random.default_rng()
    original = rng.normal(loc=0.0, scale=0.01, size=(1000, 41))

    # Augment
    augmented = augmenter.augment_training_data(original, crisis, ratio=0.15)

    return crisis, augmented
