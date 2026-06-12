"""
drl_agent.py — Deep RL Agent (Q-learning) untuk Auto Trading.
Minimal dependencies: numpy, pickle (stdlib).
"""

import numpy as np
import pickle


class CryptoPortfolioEnv:
    """
    Simple crypto trading environment with discrete actions.
    Action space: 0=HOLD, 1=BUY_20, 2=BUY_50, 3=SELL
    Observation: 8-dim vector.
    """

    HOLD = 0
    BUY_20 = 1
    BUY_50 = 2
    SELL = 3

    def __init__(self, initial_balance=100000):
        self.action_size = 4
        self.observation_size = 8
        self.initial_balance = float(initial_balance)
        self.current_step = 0
        self.position_size = 0.0
        self.balance = float(initial_balance)
        self.trade_history = []
        self.prev_portfolio_value = float(initial_balance)

    def _get_observation(self, market_state):
        """Build an 8-dim observation array from a market_state dict."""
        stress = market_state.get("stress", 0.0)
        rsi = market_state.get("rsi", 50.0)
        price = market_state.get("price", 0.0)
        momentum = market_state.get("momentum", 0.0)

        obs = np.array([
            float(stress),           # 0: current stress level
            float(rsi) / 100.0,      # 1: normalized RSI
            float(self.position_size),  # 2: position size (0..1 fraction)
            float(price) / 100000.0, # 3: normalized price
            float(momentum),         # 4: momentum indicator
            0.0,                     # 5: reserved
            0.0,                     # 6: reserved
            0.0,                     # 7: reserved
        ], dtype=np.float32)
        return obs

    def step(self, action, market_state):
        """
        Execute one step in the environment.

        Parameters
        ----------
        action : int
            0=HOLD, 1=BUY_20, 2=BUY_50, 3=SELL
        market_state : dict
            Must contain keys: stress, rsi, price, momentum

        Returns
        -------
        next_obs : np.ndarray
        reward : float
        done : bool
        info : dict
        """
        price = float(market_state.get("price", 0.0))
        stress = float(market_state.get("stress", 0.0))

        # Store price for position value calculation
        old_price = getattr(self, '_last_price', price)
        self._last_price = price

        # Portfolio value before action
        old_portfolio_value = self.balance + self.position_size * old_price

        # Execute action
        action_name = "HOLD"
        reward_penalty = 0.0
        if action == self.HOLD:
            action_name = "HOLD"
        elif action == self.BUY_20:
            action_name = "BUY_20"
            if stress < 0.7:  # only buy when stress is not extreme
                buy_amount = 0.20 * self.balance
                cost = buy_amount
                fee = cost * 0.001  # 0.1% transaction cost
                if cost + fee <= self.balance:
                    bought = cost / price if price > 0 else 0
                    self.position_size += bought
                    self.balance -= (cost + fee)
                    reward_penalty = -0.01  # fixed transaction cost penalty
                else:
                    reward_penalty = -0.02
            else:
                reward_penalty = -0.1  # penalty for buying under high stress
        elif action == self.BUY_50:
            action_name = "BUY_50"
            if stress < 0.7:
                buy_amount = 0.50 * self.balance
                cost = buy_amount
                fee = cost * 0.001
                if cost + fee <= self.balance:
                    bought = cost / price if price > 0 else 0
                    self.position_size += bought
                    self.balance -= (cost + fee)
                    reward_penalty = -0.01
                else:
                    reward_penalty = -0.02
            else:
                reward_penalty = -0.1
        elif action == self.SELL:
            action_name = "SELL"
            if self.position_size > 1e-8:
                sell_amount = self.position_size * 0.30
                proceeds = sell_amount * price
                fee = proceeds * 0.001
                self.balance += (proceeds - fee)
                self.position_size -= sell_amount
                reward_penalty = -0.01
            else:
                reward_penalty = -0.02
        else:
            action_name = "HOLD"
            reward_penalty = 0.0

        # Portfolio value after action
        new_portfolio_value = self.balance + self.position_size * price

        # Reward: change in portfolio value + penalties
        portfolio_return = (new_portfolio_value - old_portfolio_value) / max(old_portfolio_value, 1.0)
        reward = portfolio_return + reward_penalty if action not in (self.HOLD,) else portfolio_return

        info = {
            "action": action_name,
            "balance": self.balance,
            "position_size": self.position_size,
            "portfolio_value": new_portfolio_value,
            "price": price,
            "stress": stress,
            "reward_penalty": reward_penalty if action != self.HOLD else 0.0,
        }

        self.trade_history.append(info)

        # Build next observation
        next_obs = self._get_observation(market_state)
        done = False  # episodes are managed externally

        return next_obs, reward, done, info

    def reset(self, market_state=None):
        """Reset environment to initial state."""
        self.current_step = 0
        self.position_size = 0.0
        self.balance = float(self.initial_balance)
        self.trade_history = []
        self.prev_portfolio_value = float(self.initial_balance)
        self._last_price = 0.0

        if market_state is None:
            market_state = {"stress": 0.0, "rsi": 50.0, "price": 0.0, "momentum": 0.0}
        return self._get_observation(market_state)

    def get_portfolio_value(self, price):
        """Current portfolio value (balance + position value at given price)."""
        return self.balance + self.position_size * price


class QLearningAgent:
    """
    Tabular Q-learning agent with discretized state space.
    """

    def __init__(self, state_bins=10, action_size=4):
        self.state_bins = state_bins
        self.action_size = action_size
        self.learning_rate = 0.1
        self.discount = 0.95
        self.epsilon = 0.1

        # Q-table: key = discretized state tuple -> array of Q-values per action
        self.q_table = {}

    def _discretize_state(self, state):
        """
        Convert continuous state vector into a discrete tuple of bin indices.

        Parameters
        ----------
        state : np.ndarray or list
            Continuous state vector (expected length = 8).

        Returns
        -------
        tuple of int
            Bin indices for each state dimension.
        """
        state = np.asarray(state, dtype=np.float32).flatten()

        # Define ranges for each dimension (min, max)
        ranges = [
            (0.0, 1.0),     # 0: stress
            (0.0, 1.0),     # 1: rsi/100
            (0.0, 1.0),     # 2: position_size
            (0.0, 1.0),     # 3: price/100000
            (-1.0, 1.0),    # 4: momentum
            (-1.0, 1.0),    # 5: reserved
            (-1.0, 1.0),    # 6: reserved
            (-1.0, 1.0),    # 7: reserved
        ]

        bins = []
        for i, val in enumerate(state):
            lo, hi = ranges[i] if i < len(ranges) else (-1.0, 1.0)
            if hi - lo < 1e-12:
                bins.append(0)
                continue
            # Clip to range
            clipped = max(lo, min(hi, val))
            # Map to bin index
            idx = int((clipped - lo) / (hi - lo) * (self.state_bins - 1))
            idx = max(0, min(self.state_bins - 1, idx))
            bins.append(idx)

        return tuple(bins)

    def act(self, state, training=True):
        """
        Epsilon-greedy action selection.

        Parameters
        ----------
        state : np.ndarray or list
            Continuous state vector.
        training : bool
            If True, use epsilon-greedy; if False, always greedy.

        Returns
        -------
        int
            Selected action.
        """
        discrete_state = self._discretize_state(state)

        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)

        # Greedy: pick action with highest Q-value (default to 0)
        q_values = self.q_table.get(discrete_state, np.zeros(self.action_size))
        return int(np.argmax(q_values))

    def learn(self, state, action, reward, next_state, done=False):
        """
        Perform one Q-learning update.

        Q(s, a) <- Q(s, a) + lr * [r + gamma * max Q(s', a') - Q(s, a)]
        """
        discrete_state = self._discretize_state(state)
        discrete_next = self._discretize_state(next_state)

        # Initialize Q-values if unseen state
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.action_size, dtype=np.float32)
        if discrete_next not in self.q_table:
            self.q_table[discrete_next] = np.zeros(self.action_size, dtype=np.float32)

        q_current = self.q_table[discrete_state][action]
        q_next_max = np.max(self.q_table[discrete_next]) if not done else 0.0

        target = reward + self.discount * q_next_max
        self.q_table[discrete_state][action] += self.learning_rate * (target - q_current)

    def save(self, path):
        """Save Q-table to disk using pickle."""
        with open(path, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'state_bins': self.state_bins,
                'action_size': self.action_size,
                'learning_rate': self.learning_rate,
                'discount': self.discount,
                'epsilon': self.epsilon,
            }, f)

    def load(self, path):
        """Load Q-table from disk using pickle."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.q_table = data['q_table']
        self.state_bins = data.get('state_bins', self.state_bins)
        self.action_size = data.get('action_size', self.action_size)
        self.learning_rate = data.get('learning_rate', self.learning_rate)
        self.discount = data.get('discount', self.discount)
        self.epsilon = data.get('epsilon', self.epsilon)


def train_drl_agent(historical_data, episodes=500, state_bins=10):
    """
    Train a Q-learning agent on historical market data.

    Parameters
    ----------
    historical_data : list of dict
        Each dict must contain at least: stress, rsi, price, momentum.
    episodes : int
        Number of training episodes.
    state_bins : int
        Number of bins per state dimension.

    Returns
    -------
    QLearningAgent
        Trained agent.
    """
    env = CryptoPortfolioEnv(initial_balance=100000)
    agent = QLearningAgent(state_bins=state_bins, action_size=env.action_size)

    n_steps = len(historical_data)

    for ep in range(episodes):
        env.reset()
        total_reward = 0.0

        for t in range(n_steps - 1):
            market_state = historical_data[t]
            next_market_state = historical_data[t + 1]

            state = env._get_observation(market_state)
            action = agent.act(state, training=True)
            next_state, reward, done, info = env.step(action, next_market_state)
            agent.learn(state, action, reward, next_state, done=False)

            total_reward += reward

        # Decay epsilon slightly each episode
        agent.epsilon = max(0.01, agent.epsilon * 0.995)

        if (ep + 1) % 100 == 0:
            print(f"[train_drl_agent] Episode {ep + 1}/{episodes} — total_reward={total_reward:.4f}, epsilon={agent.epsilon:.4f}")

    return agent


def get_trading_signal(current_market_state, agent=None):
    """
    Get a trading signal based on the current market state.

    Parameters
    ----------
    current_market_state : dict
        Must contain keys: stress (0-1), rsi, price, momentum.
    agent : QLearningAgent or None
        If None, use a simple rule-based signal.

    Returns
    -------
    str
        One of: 'HOLD', 'BUY_20', 'BUY_50', 'SELL'
    """
    stress = float(current_market_state.get("stress", 0.5))
    rsi = float(current_market_state.get("rsi", 50.0))
    momentum = float(current_market_state.get("momentum", 0.0))
    price = float(current_market_state.get("price", 0.0))

    if agent is not None:
        # Use trained agent
        state = CryptoPortfolioEnv()._get_observation(current_market_state)
        action = agent.act(state, training=False)
        mapping = {0: "HOLD", 1: "BUY_20", 2: "BUY_50", 3: "SELL"}
        return mapping.get(action, "HOLD")

    # Rule-based signal when no agent is provided
    if stress < 0.3 and rsi < 35 and momentum > 0.05:
        return "BUY_50"
    elif stress < 0.5 and rsi < 40 and momentum > 0:
        return "BUY_20"
    elif stress > 0.75 or rsi > 75:
        return "SELL"
    else:
        return "HOLD"
