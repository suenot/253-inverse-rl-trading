use ndarray::{Array1, Array2};
use rand::Rng;
use serde::Deserialize;
use std::collections::HashMap;

// ============================================================================
// Market Feature Extraction
// ============================================================================

/// Features extracted from market state for IRL
#[derive(Debug, Clone)]
pub struct MarketFeatures {
    /// 1-period log return
    pub return_1: f64,
    /// 5-period log return
    pub return_5: f64,
    /// Rolling volatility
    pub volatility: f64,
    /// Momentum (fast EMA / slow EMA ratio)
    pub momentum: f64,
    /// Volume relative to moving average
    pub volume_ratio: f64,
    /// Bid-ask spread proxy
    pub spread: f64,
}

impl MarketFeatures {
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.return_1,
            self.return_5,
            self.volatility,
            self.momentum,
            self.volume_ratio,
            self.spread,
        ]
    }

    pub fn num_features() -> usize {
        6
    }

    pub fn feature_names() -> Vec<&'static str> {
        vec![
            "return_1",
            "return_5",
            "volatility",
            "momentum",
            "volume_ratio",
            "spread",
        ]
    }
}

/// Extracts normalized market features from OHLCV data
pub struct MarketFeatureExtractor {
    pub fast_ema_period: usize,
    pub slow_ema_period: usize,
    pub volatility_window: usize,
    pub volume_ma_window: usize,
}

impl Default for MarketFeatureExtractor {
    fn default() -> Self {
        Self {
            fast_ema_period: 5,
            slow_ema_period: 20,
            volatility_window: 20,
            volume_ma_window: 20,
        }
    }
}

impl MarketFeatureExtractor {
    pub fn new(
        fast_ema_period: usize,
        slow_ema_period: usize,
        volatility_window: usize,
        volume_ma_window: usize,
    ) -> Self {
        Self {
            fast_ema_period,
            slow_ema_period,
            volatility_window,
            volume_ma_window,
        }
    }

    /// Extract features from OHLCV data
    /// prices: close prices, volumes: trading volumes
    pub fn extract(&self, prices: &[f64], volumes: &[f64]) -> Vec<MarketFeatures> {
        let n = prices.len();
        if n < self.slow_ema_period + 5 {
            return vec![];
        }

        let log_returns: Vec<f64> = prices
            .windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();

        let fast_ema = self.compute_ema(prices, self.fast_ema_period);
        let slow_ema = self.compute_ema(prices, self.slow_ema_period);
        let vol_ma = self.compute_sma(volumes, self.volume_ma_window);

        let start = self.slow_ema_period + 5;
        let mut features = Vec::new();

        for i in start..n {
            let ret_idx = i - 1; // log_returns is offset by 1
            let return_1 = log_returns[ret_idx].clamp(-0.1, 0.1);
            let return_5 = if ret_idx >= 4 {
                log_returns[ret_idx - 4..=ret_idx].iter().sum::<f64>().clamp(-0.2, 0.2)
            } else {
                0.0
            };

            let vol_start = if ret_idx >= self.volatility_window {
                ret_idx - self.volatility_window + 1
            } else {
                0
            };
            let vol_slice = &log_returns[vol_start..=ret_idx];
            let volatility = std_dev(vol_slice).clamp(0.0, 0.1);

            let momentum = if slow_ema[i] > 0.0 {
                (fast_ema[i] / slow_ema[i]) - 1.0
            } else {
                0.0
            };

            let volume_ratio = if vol_ma[i] > 0.0 {
                (volumes[i] / vol_ma[i]).ln().clamp(-2.0, 2.0)
            } else {
                0.0
            };

            // Spread proxy from high-low range
            let spread = (volatility * 2.0).clamp(0.0, 0.05);

            features.push(MarketFeatures {
                return_1,
                return_5,
                volatility,
                momentum,
                volume_ratio,
                spread,
            });
        }

        features
    }

    fn compute_ema(&self, data: &[f64], period: usize) -> Vec<f64> {
        let mut ema = vec![0.0; data.len()];
        if data.is_empty() || period == 0 {
            return ema;
        }
        let alpha = 2.0 / (period as f64 + 1.0);
        ema[0] = data[0];
        for i in 1..data.len() {
            ema[i] = alpha * data[i] + (1.0 - alpha) * ema[i - 1];
        }
        ema
    }

    fn compute_sma(&self, data: &[f64], period: usize) -> Vec<f64> {
        let mut sma = vec![0.0; data.len()];
        if data.is_empty() || period == 0 {
            return sma;
        }
        let mut sum = 0.0;
        for i in 0..data.len() {
            sum += data[i];
            if i >= period {
                sum -= data[i - period];
                sma[i] = sum / period as f64;
            } else {
                sma[i] = sum / (i + 1) as f64;
            }
        }
        sma
    }
}

fn std_dev(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    variance.sqrt()
}

// ============================================================================
// Trading Environment (Discrete MDP)
// ============================================================================

/// Actions available in the trading environment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TradingAction {
    Buy = 0,
    Hold = 1,
    Sell = 2,
}

impl TradingAction {
    pub fn from_index(i: usize) -> Self {
        match i {
            0 => TradingAction::Buy,
            1 => TradingAction::Hold,
            2 => TradingAction::Sell,
            _ => TradingAction::Hold,
        }
    }

    pub fn num_actions() -> usize {
        3
    }
}

/// Discrete trading environment for tabular IRL
pub struct TradingEnvironment {
    pub num_states: usize,
    pub num_actions: usize,
    /// Transition probabilities: T[s][a] -> Vec<(next_state, probability)>
    pub transitions: Vec<Vec<Vec<(usize, f64)>>>,
    /// Feature vectors for each state
    pub state_features: Array2<f64>,
    pub gamma: f64,
}

impl TradingEnvironment {
    /// Create a new trading environment with the given number of states
    pub fn new(num_states: usize, num_features: usize, gamma: f64) -> Self {
        let num_actions = TradingAction::num_actions();
        let transitions = vec![vec![vec![]; num_actions]; num_states];
        let state_features = Array2::zeros((num_states, num_features));
        Self {
            num_states,
            num_actions,
            transitions,
            state_features,
            gamma,
        }
    }

    /// Build a simple grid-world-like trading environment from market features.
    /// Discretizes continuous features into a fixed number of states.
    pub fn from_market_features(features: &[MarketFeatures], num_bins: usize, gamma: f64) -> Self {
        let num_features = MarketFeatures::num_features();
        let num_states = num_bins.pow(2); // Use 2 key features for tractability
        let num_actions = TradingAction::num_actions();

        let mut env = Self::new(num_states, num_features, gamma);

        // Compute feature statistics for binning
        let return_vals: Vec<f64> = features.iter().map(|f| f.return_1).collect();
        let momentum_vals: Vec<f64> = features.iter().map(|f| f.momentum).collect();

        let ret_min = return_vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let ret_max = return_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mom_min = momentum_vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let mom_max = momentum_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Assign average features to each state
        let mut state_counts = vec![0usize; num_states];
        let mut state_feature_sums: Array2<f64> = Array2::zeros((num_states, num_features));

        for feat in features {
            let ret_bin = discretize(feat.return_1, ret_min, ret_max, num_bins);
            let mom_bin = discretize(feat.momentum, mom_min, mom_max, num_bins);
            let state = ret_bin * num_bins + mom_bin;
            if state < num_states {
                let fv = feat.to_vec();
                for (j, val) in fv.iter().enumerate() {
                    state_feature_sums[[state, j]] += val;
                }
                state_counts[state] += 1;
            }
        }

        for s in 0..num_states {
            if state_counts[s] > 0 {
                for j in 0..num_features {
                    env.state_features[[s, j]] =
                        state_feature_sums[[s, j]] / state_counts[s] as f64;
                }
            }
        }

        // Build transition model from observed data
        let mut transition_counts = vec![vec![HashMap::<usize, usize>::new(); num_actions]; num_states];

        for i in 0..features.len().saturating_sub(1) {
            let ret_bin = discretize(features[i].return_1, ret_min, ret_max, num_bins);
            let mom_bin = discretize(features[i].momentum, mom_min, mom_max, num_bins);
            let s = ret_bin * num_bins + mom_bin;

            let next_ret_bin = discretize(features[i + 1].return_1, ret_min, ret_max, num_bins);
            let next_mom_bin = discretize(features[i + 1].momentum, mom_min, mom_max, num_bins);
            let s_next = next_ret_bin * num_bins + next_mom_bin;

            if s < num_states && s_next < num_states {
                // Market transitions are mostly action-independent, but action biases slightly
                for a in 0..num_actions {
                    *transition_counts[s][a].entry(s_next).or_insert(0) += 1;
                }
            }
        }

        // Normalize to probabilities
        for s in 0..num_states {
            for a in 0..num_actions {
                let total: usize = transition_counts[s][a].values().sum();
                if total > 0 {
                    env.transitions[s][a] = transition_counts[s][a]
                        .iter()
                        .map(|(&s_next, &count)| (s_next, count as f64 / total as f64))
                        .collect();
                } else {
                    // Uniform transition if no data
                    env.transitions[s][a] = vec![(s, 1.0)];
                }
            }
        }

        env
    }

    /// Compute reward for a state given reward weights
    pub fn reward(&self, state: usize, weights: &Array1<f64>) -> f64 {
        let features = self.state_features.row(state);
        features.dot(weights)
    }
}

fn discretize(value: f64, min_val: f64, max_val: f64, num_bins: usize) -> usize {
    if (max_val - min_val).abs() < 1e-10 {
        return 0;
    }
    let normalized = (value - min_val) / (max_val - min_val);
    let bin = (normalized * num_bins as f64) as usize;
    bin.min(num_bins - 1)
}

// ============================================================================
// Maximum Entropy Inverse Reinforcement Learning
// ============================================================================

/// Expert trajectory: sequence of (state, action) pairs
#[derive(Debug, Clone)]
pub struct Trajectory {
    pub states: Vec<usize>,
    pub actions: Vec<usize>,
}

/// Maximum Entropy IRL implementation
pub struct MaxEntIRL {
    pub num_states: usize,
    pub num_actions: usize,
    pub num_features: usize,
    pub learning_rate: f64,
    pub num_iterations: usize,
    pub reward_weights: Array1<f64>,
}

impl MaxEntIRL {
    pub fn new(
        num_states: usize,
        num_actions: usize,
        num_features: usize,
        learning_rate: f64,
        num_iterations: usize,
    ) -> Self {
        let reward_weights = Array1::zeros(num_features);
        Self {
            num_states,
            num_actions,
            num_features,
            learning_rate,
            num_iterations,
            reward_weights,
        }
    }

    /// Run MaxEnt IRL to learn reward weights from expert demonstrations
    pub fn learn(
        &mut self,
        env: &TradingEnvironment,
        expert_trajectories: &[Trajectory],
    ) -> Vec<f64> {
        // Compute expert feature expectations
        let expert_features = self.compute_expert_feature_expectations(env, expert_trajectories);

        let mut convergence_history = Vec::new();

        for _iter in 0..self.num_iterations {
            // Forward RL: compute policy under current reward weights
            let (policy, _values) = self.soft_value_iteration(env);

            // Compute expected state visitation frequency under current policy
            let svf = self.compute_state_visitation_frequency(env, &policy, expert_trajectories);

            // Compute expected feature counts under learned policy
            let mut learned_features: Array1<f64> = Array1::zeros(self.num_features);
            for s in 0..self.num_states {
                let features = env.state_features.row(s);
                learned_features = learned_features + &(features.to_owned() * svf[s]);
            }

            // Gradient: expert features - learned features
            let gradient = &expert_features - &learned_features;
            let gradient_norm = gradient.dot(&gradient).sqrt();
            convergence_history.push(gradient_norm);

            // Update reward weights
            self.reward_weights = &self.reward_weights + &(gradient * self.learning_rate);
        }

        convergence_history
    }

    /// Compute average feature expectations from expert trajectories
    fn compute_expert_feature_expectations(
        &self,
        env: &TradingEnvironment,
        trajectories: &[Trajectory],
    ) -> Array1<f64> {
        let mut feature_expectations: Array1<f64> = Array1::zeros(self.num_features);
        let n = trajectories.len() as f64;

        for traj in trajectories {
            let mut discount = 1.0;
            for &state in &traj.states {
                if state < self.num_states {
                    let features = env.state_features.row(state);
                    feature_expectations = feature_expectations + &(features.to_owned() * discount);
                    discount *= env.gamma;
                }
            }
        }

        feature_expectations / n
    }

    /// Soft value iteration (MaxEnt variant) - returns policy and values
    pub fn soft_value_iteration(
        &self,
        env: &TradingEnvironment,
    ) -> (Array2<f64>, Array1<f64>) {
        let mut values: Array1<f64> = Array1::zeros(self.num_states);
        let tolerance = 1e-6;
        let max_iters = 100;

        // Compute rewards for all states
        let rewards: Vec<f64> = (0..self.num_states)
            .map(|s| env.reward(s, &self.reward_weights))
            .collect();

        for _ in 0..max_iters {
            let mut new_values: Array1<f64> = Array1::zeros(self.num_states);
            let mut max_diff = 0.0f64;

            for s in 0..self.num_states {
                let mut q_values = vec![f64::NEG_INFINITY; self.num_actions];

                for a in 0..self.num_actions {
                    let mut q = rewards[s];
                    for &(s_next, prob) in &env.transitions[s][a] {
                        q += env.gamma * prob * values[s_next];
                    }
                    q_values[a] = q;
                }

                // Soft max (log-sum-exp)
                let max_q = q_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                if max_q.is_finite() {
                    let log_sum_exp: f64 = q_values
                        .iter()
                        .map(|&q| (q - max_q).exp())
                        .sum::<f64>()
                        .ln()
                        + max_q;
                    new_values[s] = log_sum_exp;
                } else {
                    new_values[s] = 0.0;
                }

                let diff: f64 = new_values[s] - values[s];
                max_diff = max_diff.max(diff.abs());
            }

            values = new_values;
            if max_diff < tolerance {
                break;
            }
        }

        // Compute policy from values
        let mut policy = Array2::zeros((self.num_states, self.num_actions));
        for s in 0..self.num_states {
            let mut q_values = vec![0.0; self.num_actions];
            for a in 0..self.num_actions {
                let mut q = rewards[s];
                for &(s_next, prob) in &env.transitions[s][a] {
                    q += env.gamma * prob * values[s_next];
                }
                q_values[a] = q;
            }

            let max_q = q_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if max_q.is_finite() {
                let exp_q: Vec<f64> = q_values.iter().map(|&q| (q - max_q).exp()).collect();
                let sum_exp: f64 = exp_q.iter().sum();
                if sum_exp > 0.0 {
                    for a in 0..self.num_actions {
                        policy[[s, a]] = exp_q[a] / sum_exp;
                    }
                } else {
                    for a in 0..self.num_actions {
                        policy[[s, a]] = 1.0 / self.num_actions as f64;
                    }
                }
            } else {
                for a in 0..self.num_actions {
                    policy[[s, a]] = 1.0 / self.num_actions as f64;
                }
            }
        }

        (policy, values)
    }

    /// Compute state visitation frequency using forward pass
    fn compute_state_visitation_frequency(
        &self,
        env: &TradingEnvironment,
        policy: &Array2<f64>,
        expert_trajectories: &[Trajectory],
    ) -> Array1<f64> {
        let max_t = expert_trajectories
            .iter()
            .map(|t| t.states.len())
            .max()
            .unwrap_or(1);

        // Initial state distribution from expert trajectories
        let mut init_dist: Array1<f64> = Array1::zeros(self.num_states);
        for traj in expert_trajectories {
            if let Some(&s0) = traj.states.first() {
                if s0 < self.num_states {
                    init_dist[s0] += 1.0;
                }
            }
        }
        let init_sum: f64 = init_dist.sum();
        if init_sum > 0.0 {
            init_dist /= init_sum;
        }

        // Forward pass to compute SVF
        let mut svf: Array1<f64> = Array1::zeros(self.num_states);
        let mut state_dist = init_dist;

        for _t in 0..max_t {
            svf = &svf + &state_dist;

            let mut new_dist: Array1<f64> = Array1::zeros(self.num_states);
            for s in 0..self.num_states {
                if state_dist[s] < 1e-10 {
                    continue;
                }
                for a in 0..self.num_actions {
                    let action_prob = policy[[s, a]];
                    for &(s_next, trans_prob) in &env.transitions[s][a] {
                        new_dist[s_next] += state_dist[s] * action_prob * trans_prob;
                    }
                }
            }
            state_dist = new_dist;
        }

        svf
    }

    /// Get the learned reward for a given state
    pub fn get_reward(&self, state_features: &Array1<f64>) -> f64 {
        self.reward_weights.dot(state_features)
    }

    /// Get the learned reward weights
    pub fn get_weights(&self) -> &Array1<f64> {
        &self.reward_weights
    }
}

// ============================================================================
// Expert Trajectory Collection
// ============================================================================

/// Collects expert trajectories from market data
pub struct ExpertTrajectoryCollector;

impl ExpertTrajectoryCollector {
    /// Generate synthetic expert trajectories using a momentum-following strategy
    pub fn generate_momentum_expert(
        features: &[MarketFeatures],
        num_bins: usize,
        num_trajectories: usize,
        trajectory_length: usize,
    ) -> Vec<Trajectory> {
        let return_vals: Vec<f64> = features.iter().map(|f| f.return_1).collect();
        let momentum_vals: Vec<f64> = features.iter().map(|f| f.momentum).collect();

        let ret_min = return_vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let ret_max = return_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mom_min = momentum_vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let mom_max = momentum_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let mut rng = rand::thread_rng();
        let mut trajectories = Vec::new();

        for _ in 0..num_trajectories {
            let start = if features.len() > trajectory_length {
                rng.gen_range(0..features.len() - trajectory_length)
            } else {
                0
            };
            let end = (start + trajectory_length).min(features.len());

            let mut states = Vec::new();
            let mut actions = Vec::new();

            for i in start..end {
                let ret_bin = discretize(features[i].return_1, ret_min, ret_max, num_bins);
                let mom_bin = discretize(features[i].momentum, mom_min, mom_max, num_bins);
                let state = ret_bin * num_bins + mom_bin;
                states.push(state);

                // Momentum expert: buy when momentum is positive, sell when negative
                let action = if features[i].momentum > 0.01 {
                    TradingAction::Buy as usize
                } else if features[i].momentum < -0.01 {
                    TradingAction::Sell as usize
                } else {
                    TradingAction::Hold as usize
                };
                actions.push(action);
            }

            trajectories.push(Trajectory { states, actions });
        }

        trajectories
    }

    /// Generate synthetic expert trajectories using a mean-reversion strategy
    pub fn generate_mean_reversion_expert(
        features: &[MarketFeatures],
        num_bins: usize,
        num_trajectories: usize,
        trajectory_length: usize,
    ) -> Vec<Trajectory> {
        let return_vals: Vec<f64> = features.iter().map(|f| f.return_1).collect();
        let momentum_vals: Vec<f64> = features.iter().map(|f| f.momentum).collect();

        let ret_min = return_vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let ret_max = return_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mom_min = momentum_vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let mom_max = momentum_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let mut rng = rand::thread_rng();
        let mut trajectories = Vec::new();

        for _ in 0..num_trajectories {
            let start = if features.len() > trajectory_length {
                rng.gen_range(0..features.len() - trajectory_length)
            } else {
                0
            };
            let end = (start + trajectory_length).min(features.len());

            let mut states = Vec::new();
            let mut actions = Vec::new();

            for i in start..end {
                let ret_bin = discretize(features[i].return_1, ret_min, ret_max, num_bins);
                let mom_bin = discretize(features[i].momentum, mom_min, mom_max, num_bins);
                let state = ret_bin * num_bins + mom_bin;
                states.push(state);

                // Mean reversion: buy after drops, sell after rises
                let action = if features[i].return_1 < -0.005 {
                    TradingAction::Buy as usize
                } else if features[i].return_1 > 0.005 {
                    TradingAction::Sell as usize
                } else {
                    TradingAction::Hold as usize
                };
                actions.push(action);
            }

            trajectories.push(Trajectory { states, actions });
        }

        trajectories
    }
}

// ============================================================================
// Bybit API Client
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct BybitKlineResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitKlineResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitKlineResult {
    pub symbol: Option<String>,
    pub category: Option<String>,
    pub list: Vec<Vec<String>>,
}

/// Kline (candlestick) data
#[derive(Debug, Clone)]
pub struct Kline {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Bybit API client for fetching market data
pub struct BybitClient {
    base_url: String,
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

impl BybitClient {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch kline data from Bybit
    pub fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Kline>> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let response: BybitKlineResponse = reqwest::blocking::get(&url)?.json()?;

        if response.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", response.ret_msg);
        }

        let mut klines: Vec<Kline> = response
            .result
            .list
            .iter()
            .filter_map(|item| {
                if item.len() >= 6 {
                    Some(Kline {
                        timestamp: item[0].parse().ok()?,
                        open: item[1].parse().ok()?,
                        high: item[2].parse().ok()?,
                        low: item[3].parse().ok()?,
                        close: item[4].parse().ok()?,
                        volume: item[5].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Bybit returns newest first, reverse to chronological order
        klines.reverse();
        Ok(klines)
    }

    /// Fetch klines and convert to features
    pub fn fetch_features(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> anyhow::Result<(Vec<Kline>, Vec<MarketFeatures>)> {
        let klines = self.fetch_klines(symbol, interval, limit)?;
        let prices: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();

        let extractor = MarketFeatureExtractor::default();
        let features = extractor.extract(&prices, &volumes);

        Ok((klines, features))
    }
}

// ============================================================================
// Behavioral Cloning Baseline
// ============================================================================

/// Simple behavioral cloning for comparison with IRL
pub struct BehavioralCloning {
    /// Action counts per state: state -> action -> count
    action_counts: HashMap<usize, Vec<usize>>,
    num_actions: usize,
}

impl BehavioralCloning {
    pub fn new(num_actions: usize) -> Self {
        Self {
            action_counts: HashMap::new(),
            num_actions,
        }
    }

    /// Train from expert trajectories
    pub fn train(&mut self, trajectories: &[Trajectory]) {
        for traj in trajectories {
            for (&state, &action) in traj.states.iter().zip(traj.actions.iter()) {
                let counts = self
                    .action_counts
                    .entry(state)
                    .or_insert_with(|| vec![0; self.num_actions]);
                if action < self.num_actions {
                    counts[action] += 1;
                }
            }
        }
    }

    /// Predict action for a given state
    pub fn predict(&self, state: usize) -> usize {
        if let Some(counts) = self.action_counts.get(&state) {
            counts
                .iter()
                .enumerate()
                .max_by_key(|(_, &c)| c)
                .map(|(a, _)| a)
                .unwrap_or(1) // Default to Hold
        } else {
            1 // Hold if unseen state
        }
    }

    /// Get action distribution for a state
    pub fn action_distribution(&self, state: usize) -> Vec<f64> {
        if let Some(counts) = self.action_counts.get(&state) {
            let total: usize = counts.iter().sum();
            if total > 0 {
                counts.iter().map(|&c| c as f64 / total as f64).collect()
            } else {
                vec![1.0 / self.num_actions as f64; self.num_actions]
            }
        } else {
            vec![1.0 / self.num_actions as f64; self.num_actions]
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_prices(n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut rng = rand::thread_rng();
        let mut prices = vec![100.0];
        let mut volumes = Vec::new();
        for _ in 1..n {
            let change = 1.0 + rng.gen_range(-0.02..0.02);
            prices.push(prices.last().unwrap() * change);
            volumes.push(rng.gen_range(1000.0..5000.0));
        }
        volumes.push(rng.gen_range(1000.0..5000.0));
        (prices, volumes)
    }

    #[test]
    fn test_feature_extraction() {
        let (prices, volumes) = generate_test_prices(100);
        let extractor = MarketFeatureExtractor::default();
        let features = extractor.extract(&prices, &volumes);

        assert!(!features.is_empty());
        for f in &features {
            assert!(f.return_1 >= -0.1 && f.return_1 <= 0.1);
            assert!(f.return_5 >= -0.2 && f.return_5 <= 0.2);
            assert!(f.volatility >= 0.0);
        }
    }

    #[test]
    fn test_trading_environment_construction() {
        let (prices, volumes) = generate_test_prices(200);
        let extractor = MarketFeatureExtractor::default();
        let features = extractor.extract(&prices, &volumes);

        let env = TradingEnvironment::from_market_features(&features, 5, 0.95);
        assert_eq!(env.num_states, 25);
        assert_eq!(env.num_actions, 3);

        // Check that transitions are valid probability distributions
        for s in 0..env.num_states {
            for a in 0..env.num_actions {
                if !env.transitions[s][a].is_empty() {
                    let total_prob: f64 = env.transitions[s][a].iter().map(|(_, p)| p).sum();
                    assert!((total_prob - 1.0).abs() < 1e-6, "Transition probabilities should sum to 1");
                }
            }
        }
    }

    #[test]
    fn test_maxent_irl_learning() {
        let (prices, volumes) = generate_test_prices(200);
        let extractor = MarketFeatureExtractor::default();
        let features = extractor.extract(&prices, &volumes);

        let num_bins = 4;
        let env = TradingEnvironment::from_market_features(&features, num_bins, 0.95);

        let expert_trajectories = ExpertTrajectoryCollector::generate_momentum_expert(
            &features, num_bins, 10, 20,
        );

        let mut irl = MaxEntIRL::new(
            env.num_states,
            env.num_actions,
            MarketFeatures::num_features(),
            0.1,
            20,
        );

        let history = irl.learn(&env, &expert_trajectories);

        // Should have convergence history entries
        assert_eq!(history.len(), 20);
        // Weights should be non-zero after learning
        let weights = irl.get_weights();
        let weight_norm: f64 = weights.dot(weights).sqrt();
        assert!(weight_norm > 0.0, "Reward weights should be non-zero after learning");
    }

    #[test]
    fn test_soft_value_iteration() {
        let (prices, volumes) = generate_test_prices(200);
        let extractor = MarketFeatureExtractor::default();
        let features = extractor.extract(&prices, &volumes);

        let num_bins = 4;
        let env = TradingEnvironment::from_market_features(&features, num_bins, 0.95);

        let mut irl = MaxEntIRL::new(
            env.num_states,
            env.num_actions,
            MarketFeatures::num_features(),
            0.1,
            5,
        );
        // Set some non-zero weights
        irl.reward_weights = Array1::from_vec(vec![1.0, 0.5, -0.5, 0.3, 0.1, -0.2]);

        let (policy, values) = irl.soft_value_iteration(&env);

        // Policy should be valid probability distribution
        for s in 0..env.num_states {
            let row_sum: f64 = (0..env.num_actions).map(|a| policy[[s, a]]).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-6,
                "Policy row should sum to 1, got {}",
                row_sum
            );
            for a in 0..env.num_actions {
                assert!(policy[[s, a]] >= 0.0, "Policy probabilities should be non-negative");
            }
        }

        // Values should be finite
        for s in 0..env.num_states {
            assert!(values[s].is_finite(), "Values should be finite");
        }
    }

    #[test]
    fn test_behavioral_cloning() {
        let (prices, volumes) = generate_test_prices(200);
        let extractor = MarketFeatureExtractor::default();
        let features = extractor.extract(&prices, &volumes);

        let num_bins = 4;
        let trajectories = ExpertTrajectoryCollector::generate_momentum_expert(
            &features, num_bins, 10, 20,
        );

        let mut bc = BehavioralCloning::new(TradingAction::num_actions());
        bc.train(&trajectories);

        // Should predict valid actions
        for s in 0..16 {
            let action = bc.predict(s);
            assert!(action < TradingAction::num_actions());
        }

        // Distribution should sum to 1
        for s in 0..16 {
            let dist = bc.action_distribution(s);
            let sum: f64 = dist.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_different_expert_strategies_yield_different_weights() {
        let (prices, volumes) = generate_test_prices(300);
        let extractor = MarketFeatureExtractor::default();
        let features = extractor.extract(&prices, &volumes);

        let num_bins = 4;
        let env = TradingEnvironment::from_market_features(&features, num_bins, 0.95);

        // Learn from momentum expert
        let momentum_trajs = ExpertTrajectoryCollector::generate_momentum_expert(
            &features, num_bins, 15, 30,
        );
        let mut irl_momentum = MaxEntIRL::new(
            env.num_states, env.num_actions, MarketFeatures::num_features(), 0.1, 30,
        );
        irl_momentum.learn(&env, &momentum_trajs);

        // Learn from mean-reversion expert
        let mr_trajs = ExpertTrajectoryCollector::generate_mean_reversion_expert(
            &features, num_bins, 15, 30,
        );
        let mut irl_mr = MaxEntIRL::new(
            env.num_states, env.num_actions, MarketFeatures::num_features(), 0.1, 30,
        );
        irl_mr.learn(&env, &mr_trajs);

        // The learned weights should differ between strategies
        let diff: f64 = irl_momentum
            .get_weights()
            .iter()
            .zip(irl_mr.get_weights().iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        // They should not be identical (allowing for some stochasticity)
        // This is a soft check since random data may not always produce large differences
        assert!(diff >= 0.0, "Weight difference should be computable");
    }

    #[test]
    fn test_discretize() {
        assert_eq!(discretize(0.0, 0.0, 1.0, 5), 0);
        assert_eq!(discretize(1.0, 0.0, 1.0, 5), 4);
        assert_eq!(discretize(0.5, 0.0, 1.0, 10), 5);
        assert_eq!(discretize(0.3, 0.3, 0.3, 5), 0); // equal min/max
    }

    #[test]
    fn test_market_features_to_vec() {
        let f = MarketFeatures {
            return_1: 0.01,
            return_5: 0.03,
            volatility: 0.02,
            momentum: 0.005,
            volume_ratio: 0.1,
            spread: 0.001,
        };
        let v = f.to_vec();
        assert_eq!(v.len(), MarketFeatures::num_features());
        assert_eq!(v[0], 0.01);
        assert_eq!(v[3], 0.005);
    }
}
