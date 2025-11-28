# Chapter 305: Inverse Reinforcement Learning for Trading

## Introduction

Standard reinforcement learning (RL) assumes a well-defined reward function is given to the agent. In trading, however, defining the right reward signal is notoriously difficult. Should we optimize for raw PnL, Sharpe ratio, maximum drawdown, or some complex multi-objective combination? The answer is often hidden in the behavior of successful traders who have internalized years of market intuition into their decision-making process.

**Inverse Reinforcement Learning (IRL)** flips the RL problem on its head: instead of learning a policy from a reward function, IRL recovers the reward function from observed expert behavior. Given a set of demonstrated trajectories from an expert trader, IRL infers *what* the expert is optimizing for, not just *how* they trade.

This chapter explores how IRL can be applied to financial markets to reverse-engineer the implicit objectives of institutional traders, market makers, and other sophisticated market participants. We implement a Maximum Entropy IRL framework in Rust with integration to Bybit exchange data, enabling practitioners to learn reward functions from real trading behavior.

### Why Inverse RL for Trading?

Traditional approaches to algorithmic trading either:
1. **Hand-craft reward functions** -- which may miss subtle objectives that expert traders implicitly optimize
2. **Use behavioral cloning** -- which copies actions but fails to generalize when market conditions change
3. **Apply standard RL** -- which requires extensive reward engineering and may overfit to the chosen metric

IRL addresses these limitations by recovering a transferable reward function that explains expert behavior. Once learned, this reward function can be used to train new policies that generalize to unseen market conditions, because the agent understands *why* certain actions are optimal, not just *what* actions to take.

## Mathematical Foundations

### The IRL Problem Formulation

We model trading as a Markov Decision Process (MDP) defined by the tuple $(S, A, T, \gamma, R)$:

- $S$: State space (market features: prices, volumes, indicators, portfolio state)
- $A$: Action space (buy, sell, hold, position sizing)
- $T(s' | s, a)$: Transition dynamics (market evolution)
- $\gamma$: Discount factor
- $R(s, a)$: Reward function (unknown, to be recovered)

Given expert demonstrations $\mathcal{D} = \{\tau_1, \tau_2, \ldots, \tau_N\}$ where each trajectory $\tau_i = (s_0, a_0, s_1, a_1, \ldots)$, the goal is to find $R^*$ such that the expert's policy $\pi_E$ is optimal under $R^*$.

### Feature Expectation Matching

The foundational IRL insight (Abbeel & Ng, 2004) is that the reward function can be expressed as a linear combination of state features:

$$R(s) = \boldsymbol{w}^T \boldsymbol{\phi}(s)$$

where $\boldsymbol{\phi}(s) \in \mathbb{R}^k$ is a feature vector and $\boldsymbol{w} \in \mathbb{R}^k$ are the reward weights to be learned.

The **feature expectation** of a policy $\pi$ is:

$$\boldsymbol{\mu}(\pi) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t \boldsymbol{\phi}(s_t) \mid \pi\right]$$

The IRL constraint requires that the expert's feature expectations are at least as good as any other policy's:

$$\boldsymbol{w}^T \boldsymbol{\mu}(\pi_E) \geq \boldsymbol{w}^T \boldsymbol{\mu}(\pi) \quad \forall \pi$$

### Maximum Entropy IRL (Ziebart et al., 2008)

The Maximum Entropy (MaxEnt) IRL framework resolves the ambiguity in IRL by choosing the distribution over trajectories with the highest entropy, subject to matching the expert's feature expectations. This yields:

$$P(\tau | \boldsymbol{w}) = \frac{1}{Z(\boldsymbol{w})} \exp\left(\boldsymbol{w}^T \boldsymbol{f}_\tau\right)$$

where $\boldsymbol{f}_\tau = \sum_{t} \boldsymbol{\phi}(s_t)$ is the cumulative feature count along trajectory $\tau$, and $Z(\boldsymbol{w})$ is the partition function.

The log-likelihood of the expert demonstrations under this model is:

$$\mathcal{L}(\boldsymbol{w}) = \sum_{i=1}^{N} \boldsymbol{w}^T \boldsymbol{f}_{\tau_i} - N \log Z(\boldsymbol{w})$$

The gradient of the log-likelihood is:

$$\nabla_{\boldsymbol{w}} \mathcal{L} = \boldsymbol{f}_{\text{expert}} - \mathbb{E}_{\pi_{\boldsymbol{w}}}[\boldsymbol{f}]$$

where $\boldsymbol{f}_{\text{expert}} = \frac{1}{N}\sum_i \boldsymbol{f}_{\tau_i}$ is the average expert feature count, and $\mathbb{E}_{\pi_{\boldsymbol{w}}}[\boldsymbol{f}]$ is the expected feature count under the current policy induced by the learned reward. The gradient ascent update is:

$$\boldsymbol{w} \leftarrow \boldsymbol{w} + \alpha \left(\boldsymbol{f}_{\text{expert}} - \mathbb{E}_{\pi_{\boldsymbol{w}}}[\boldsymbol{f}]\right)$$

This iterative process alternates between:
1. **Forward RL**: Solve for the optimal policy $\pi_{\boldsymbol{w}}$ under current reward weights $\boldsymbol{w}$
2. **Backward gradient**: Update $\boldsymbol{w}$ to move the expected feature counts toward the expert's

### Value Iteration with Linear Reward

Given learned reward weights $\boldsymbol{w}$, the reward at state $s$ is $R(s) = \boldsymbol{w}^T \boldsymbol{\phi}(s)$. Standard value iteration computes:

$$V(s) = \max_a \left[ R(s) + \gamma \sum_{s'} T(s'|s,a) V(s') \right]$$

$$Q(s, a) = R(s) + \gamma \sum_{s'} T(s'|s,a) V(s')$$

The soft (MaxEnt) variant uses a softmax instead of max:

$$V_{\text{soft}}(s) = \text{softmax}_a \left[ R(s) + \gamma \sum_{s'} T(s'|s,a) V_{\text{soft}}(s') \right]$$

which yields a stochastic policy $\pi(a|s) \propto \exp(Q(s,a) - V(s))$.

## Applications in Trading

### Reverse-Engineering Institutional Trader Objectives

Institutional traders often have complex, multi-faceted objectives that go beyond simple profit maximization:

- **Execution quality**: Minimizing market impact while completing large orders
- **Risk constraints**: Maintaining portfolio VaR within limits
- **Regulatory compliance**: Ensuring best execution obligations are met
- **Information leakage**: Preventing other participants from detecting their intent

By observing their order flow patterns (via public trade and order book data), IRL can infer the relative importance they place on each of these objectives. The recovered reward weights $\boldsymbol{w}$ reveal whether a particular institutional participant prioritizes speed over impact, or vice versa.

### Market Maker Incentive Analysis

Market makers provide liquidity by continuously quoting bid-ask spreads. Their implicit reward function balances:

- **Spread capture**: Earning the bid-ask spread on round-trip trades
- **Inventory risk**: Avoiding excessive directional exposure
- **Adverse selection**: Minimizing losses to informed traders
- **Rebate optimization**: Maximizing exchange fee rebates

IRL applied to market maker behavior can reveal how these incentives shift across different market regimes (high volatility, low liquidity, around news events). This information is valuable for designing competing strategies or for exchanges optimizing their fee structures.

### Expert Strategy Decomposition

Given a profitable trader's history, IRL can decompose their strategy into interpretable reward components:

- **Momentum weight**: How much the expert rewards trend-following features
- **Mean reversion weight**: The importance of reversion signals
- **Volatility preference**: Whether the expert seeks or avoids volatile conditions
- **Correlation sensitivity**: How cross-asset correlations influence decisions

## Rust Implementation

Our implementation in `rust/src/lib.rs` provides:

1. **`MarketFeatureExtractor`**: Extracts normalized features from raw market data including returns, volatility, momentum, volume ratio, and spread. These form the feature vector $\boldsymbol{\phi}(s)$.

2. **`MaxEntIRL`**: The core IRL algorithm implementing:
   - Gradient ascent on log-likelihood to learn reward weights $\boldsymbol{w}$
   - Soft value iteration with the learned reward
   - Feature expectation computation from both expert trajectories and learned policy
   - Convergence monitoring via feature count difference

3. **`TradingEnvironment`**: A discrete MDP environment for trading with configurable states and actions, supporting transition dynamics and reward computation.

4. **`ExpertTrajectoryCollector`**: Generates or collects expert demonstrations, either from simulated experts or from real trading data.

5. **`BybitClient`**: Fetches real-time and historical market data from Bybit's API, converting raw kline data into feature vectors suitable for IRL.

### Feature Engineering for IRL

The feature vector $\boldsymbol{\phi}(s)$ for each market state includes:

| Feature | Description | Normalization |
|---------|-------------|---------------|
| `return_1` | 1-period log return | Clipped to [-0.1, 0.1] |
| `return_5` | 5-period log return | Clipped to [-0.2, 0.2] |
| `volatility` | Rolling std of returns | Min-max scaled |
| `momentum` | EMA ratio (fast/slow) | Centered at 1.0 |
| `volume_ratio` | Volume vs moving average | Log-scaled |
| `spread` | Bid-ask spread proxy | Min-max scaled |

## Bybit Data Integration

The implementation connects to Bybit's public API to fetch historical kline (candlestick) data:

```
GET https://api.bybit.com/v5/market/kline?category=linear&symbol=BTCUSDT&interval=15&limit=200
```

The raw OHLCV data is transformed into the feature representation described above. Expert trajectories are constructed by labeling historical periods where specific trading patterns (e.g., momentum-following, mean-reversion) are observed, then running IRL to recover the implicit reward function driving those patterns.

### Data Pipeline

1. **Fetch**: Pull BTCUSDT kline data from Bybit
2. **Transform**: Compute technical features (returns, volatility, momentum, volume ratio)
3. **Discretize**: Map continuous features to discrete states for tabular IRL
4. **Label**: Identify expert actions from price movements and volume patterns
5. **Learn**: Run MaxEnt IRL to recover reward weights
6. **Evaluate**: Compare learned policy against behavioral cloning baseline

## Key Takeaways

1. **IRL recovers "why" not just "what"**: Unlike behavioral cloning which copies actions, IRL infers the underlying reward function, producing policies that generalize to new market conditions.

2. **MaxEnt IRL provides principled ambiguity resolution**: The maximum entropy framework selects the least committed reward function consistent with expert behavior, avoiding overfitting to specific trajectories.

3. **Feature engineering is critical**: The quality of recovered reward functions depends heavily on the feature representation $\boldsymbol{\phi}(s)$. Domain knowledge about market microstructure should inform feature design.

4. **Institutional behavior reveals hidden objectives**: Applying IRL to order flow data can uncover the multi-faceted objectives of institutional participants, providing strategic intelligence.

5. **Learned rewards transfer across markets**: Reward functions recovered from one market can potentially be applied to similar instruments, enabling transfer learning across assets.

6. **Computational cost is manageable in discrete settings**: Tabular MaxEnt IRL with soft value iteration is tractable for discretized trading environments, though continuous state spaces require function approximation extensions.

7. **Combining IRL with forward RL is powerful**: The two-step approach of (1) learning rewards via IRL, then (2) optimizing a policy via standard RL with the learned reward often outperforms either approach alone.

## References

- Ng, A. & Russell, S. (2000). Algorithms for Inverse Reinforcement Learning. *ICML*.
- Abbeel, P. & Ng, A. (2004). Apprenticeship Learning via Inverse Reinforcement Learning. *ICML*.
- Ziebart, B. et al. (2008). Maximum Entropy Inverse Reinforcement Learning. *AAAI*.
- Wulfmeier, M. et al. (2015). Maximum Entropy Deep Inverse Reinforcement Learning. *arXiv*.
- Yang, S. et al. (2020). Inverse Reinforcement Learning for Order Execution. *ICAIF*.
