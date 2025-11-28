use inverse_rl_trading::*;
use ndarray::Array1;

fn main() -> anyhow::Result<()> {
    println!("=== Chapter 305: Inverse RL Trading ===\n");

    // -----------------------------------------------------------------------
    // Step 1: Fetch BTCUSDT data from Bybit
    // -----------------------------------------------------------------------
    println!("Step 1: Fetching BTCUSDT kline data from Bybit...");
    let client = BybitClient::new();

    let (klines, features) = match client.fetch_features("BTCUSDT", "15", 200) {
        Ok(result) => {
            println!("  Fetched {} klines, extracted {} feature vectors", result.0.len(), result.1.len());
            if let Some(last) = result.0.last() {
                println!("  Latest close price: {:.2}", last.close);
            }
            result
        }
        Err(e) => {
            println!("  Could not fetch from Bybit ({}), generating synthetic data...", e);
            generate_synthetic_data()
        }
    };

    if features.len() < 30 {
        println!("  Insufficient features, generating synthetic data...");
        let (klines, features) = generate_synthetic_data();
        return run_irl_pipeline(&klines, &features);
    }

    run_irl_pipeline(&klines, &features)
}

fn generate_synthetic_data() -> (Vec<Kline>, Vec<MarketFeatures>) {
    let mut rng = rand::thread_rng();
    use rand::Rng;

    let n = 200;
    let mut prices = vec![50000.0];
    let mut volumes = Vec::new();
    let mut klines = Vec::new();

    for i in 0..n {
        let change = 1.0 + rng.gen_range(-0.015..0.015);
        let close = prices.last().unwrap() * change;
        prices.push(close);
        let vol = rng.gen_range(100.0..1000.0);
        volumes.push(vol);
        klines.push(Kline {
            timestamp: 1700000000000 + (i as u64 * 900000),
            open: prices[i],
            high: prices[i].max(close) * (1.0 + rng.gen_range(0.0..0.005)),
            low: prices[i].min(close) * (1.0 - rng.gen_range(0.0..0.005)),
            close,
            volume: vol,
        });
    }
    volumes.push(rng.gen_range(100.0..1000.0));

    let extractor = MarketFeatureExtractor::default();
    let features = extractor.extract(&prices[..prices.len() - 1], &volumes[..volumes.len() - 1]);
    println!("  Generated {} synthetic klines, {} features", klines.len(), features.len());

    (klines, features)
}

fn run_irl_pipeline(_klines: &[Kline], features: &[MarketFeatures]) -> anyhow::Result<()> {
    let num_bins = 5;
    let num_states = num_bins * num_bins;
    let num_features = MarketFeatures::num_features();

    // -----------------------------------------------------------------------
    // Step 2: Build trading environment
    // -----------------------------------------------------------------------
    println!("\nStep 2: Building discrete trading environment...");
    let env = TradingEnvironment::from_market_features(features, num_bins, 0.95);
    println!("  States: {}, Actions: {}, Features: {}", env.num_states, env.num_actions, num_features);

    // Count non-empty transitions
    let non_empty: usize = env.transitions.iter()
        .flat_map(|s| s.iter())
        .filter(|t| !t.is_empty())
        .count();
    println!("  Non-empty transition entries: {}", non_empty);

    // -----------------------------------------------------------------------
    // Step 3: Generate expert demonstrations
    // -----------------------------------------------------------------------
    println!("\nStep 3: Simulating expert trader demonstrations...");

    println!("\n  --- Momentum Expert ---");
    let momentum_trajs = ExpertTrajectoryCollector::generate_momentum_expert(
        features, num_bins, 20, 30,
    );
    println!("  Generated {} momentum expert trajectories", momentum_trajs.len());
    print_trajectory_stats(&momentum_trajs);

    println!("\n  --- Mean Reversion Expert ---");
    let mr_trajs = ExpertTrajectoryCollector::generate_mean_reversion_expert(
        features, num_bins, 20, 30,
    );
    println!("  Generated {} mean-reversion expert trajectories", mr_trajs.len());
    print_trajectory_stats(&mr_trajs);

    // -----------------------------------------------------------------------
    // Step 4: Run MaxEnt IRL to recover reward weights
    // -----------------------------------------------------------------------
    println!("\nStep 4: Running MaxEnt IRL to recover reward weights...");

    println!("\n  --- Learning from Momentum Expert ---");
    let mut irl_momentum = MaxEntIRL::new(
        num_states,
        TradingAction::num_actions(),
        num_features,
        0.05,
        50,
    );
    let history_mom = irl_momentum.learn(&env, &momentum_trajs);
    println!("  Convergence (first 5): {:?}", &history_mom[..5.min(history_mom.len())]);
    println!("  Convergence (last 5):  {:?}", &history_mom[history_mom.len().saturating_sub(5)..]);
    print_reward_weights("Momentum", irl_momentum.get_weights());

    println!("\n  --- Learning from Mean Reversion Expert ---");
    let mut irl_mr = MaxEntIRL::new(
        num_states,
        TradingAction::num_actions(),
        num_features,
        0.05,
        50,
    );
    let history_mr = irl_mr.learn(&env, &mr_trajs);
    println!("  Convergence (first 5): {:?}", &history_mr[..5.min(history_mr.len())]);
    println!("  Convergence (last 5):  {:?}", &history_mr[history_mr.len().saturating_sub(5)..]);
    print_reward_weights("Mean Reversion", irl_mr.get_weights());

    // -----------------------------------------------------------------------
    // Step 5: Compare IRL policy with Behavioral Cloning
    // -----------------------------------------------------------------------
    println!("\nStep 5: Comparing IRL policy with Behavioral Cloning...");

    // Get IRL policy
    let (irl_policy, _irl_values) = irl_momentum.soft_value_iteration(&env);

    // Train behavioral cloning
    let mut bc = BehavioralCloning::new(TradingAction::num_actions());
    bc.train(&momentum_trajs);

    println!("\n  State | IRL Policy (B/H/S)       | BC Policy (B/H/S)");
    println!("  ------|--------------------------|-------------------------");

    for s in 0..num_states.min(10) {
        let irl_dist: Vec<f64> = (0..TradingAction::num_actions())
            .map(|a| irl_policy[[s, a]])
            .collect();
        let bc_dist = bc.action_distribution(s);

        println!(
            "  {:>5} | {:.3} / {:.3} / {:.3}       | {:.3} / {:.3} / {:.3}",
            s,
            irl_dist[0], irl_dist[1], irl_dist[2],
            bc_dist[0], bc_dist[1], bc_dist[2],
        );
    }

    // -----------------------------------------------------------------------
    // Step 6: Evaluate on held-out data
    // -----------------------------------------------------------------------
    println!("\nStep 6: Policy evaluation summary...");

    // Compare policy entropy (IRL should be more stochastic = higher entropy)
    let mut irl_entropy = 0.0;
    let mut bc_entropy = 0.0;

    for s in 0..num_states {
        for a in 0..TradingAction::num_actions() {
            let irl_p = irl_policy[[s, a]];
            if irl_p > 1e-10 {
                irl_entropy -= irl_p * irl_p.ln();
            }
            let bc_p = bc.action_distribution(s)[a];
            if bc_p > 1e-10 {
                bc_entropy -= bc_p * bc_p.ln();
            }
        }
    }
    irl_entropy /= num_states as f64;
    bc_entropy /= num_states as f64;

    println!("  Average policy entropy:");
    println!("    IRL:                {:.4}", irl_entropy);
    println!("    Behavioral Cloning: {:.4}", bc_entropy);
    println!("  (Higher entropy = more exploratory policy)");

    // Compare weight interpretations
    println!("\nStep 7: Reward function interpretation...");
    let names = MarketFeatures::feature_names();
    let mom_w = irl_momentum.get_weights();
    let mr_w = irl_mr.get_weights();

    println!("\n  Feature           | Momentum w | MeanRev w  | Interpretation");
    println!("  ------------------|------------|------------|----------------------------");
    for (i, name) in names.iter().enumerate() {
        let interp = if (mom_w[i] - mr_w[i]).abs() > 0.1 {
            "Differs between strategies"
        } else {
            "Similar across strategies"
        };
        println!(
            "  {:>17} | {:>10.4} | {:>10.4} | {}",
            name, mom_w[i], mr_w[i], interp
        );
    }

    println!("\n=== Inverse RL Trading Complete ===");
    println!("Key insight: IRL recovers different reward weights for different");
    println!("expert strategies, revealing what each expert implicitly optimizes.");

    Ok(())
}

fn print_trajectory_stats(trajectories: &[Trajectory]) {
    let total_steps: usize = trajectories.iter().map(|t| t.actions.len()).sum();
    let mut action_counts = [0usize; 3];
    for traj in trajectories {
        for &a in &traj.actions {
            if a < 3 {
                action_counts[a] += 1;
            }
        }
    }
    println!("  Total steps: {}", total_steps);
    println!(
        "  Action distribution: Buy={:.1}%, Hold={:.1}%, Sell={:.1}%",
        action_counts[0] as f64 / total_steps as f64 * 100.0,
        action_counts[1] as f64 / total_steps as f64 * 100.0,
        action_counts[2] as f64 / total_steps as f64 * 100.0,
    );
}

fn print_reward_weights(label: &str, weights: &Array1<f64>) {
    let names = MarketFeatures::feature_names();
    println!("  Learned reward weights ({}):", label);
    for (i, name) in names.iter().enumerate() {
        let bar_len = (weights[i].abs() * 20.0).min(30.0) as usize;
        let bar = if weights[i] >= 0.0 {
            format!("+{}", "#".repeat(bar_len))
        } else {
            format!("-{}", "#".repeat(bar_len))
        };
        println!("    {:>15}: {:>8.4} {}", name, weights[i], bar);
    }
}
