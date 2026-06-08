# INTEGRATION.md — wasserstein-agents-rs × spectral-fleet-rs × categorical-agents-rs

**Wasserstein agents** coordinate multi-agent fleets via optimal transport,
earth-mover distances, and JKO gradient flows. They connect to spectral
methods for embedding distributions and to category theory for composing
transport plans monadically.

## Synergy Map

```
spectral-fleet-rs              wasserstein-agents-rs          categorical-agents-rs
┌──────────────────┐           ┌──────────────────────┐       ┌─────────────────────┐
│ l2_norm           │◄─────────►│ AgentDistribution    │◄─────►│ StateMonad          │
│ normalize         │           │ SinkhornSolver       │       │ ListMonad           │
│ SpectralCluster   │           │ OptimalTransport     │       │ MaybeMonad          │
│ kmeans            │           │ JKOScheme            │       │ Adjunction          │
└──────────────────┘           └──────────────────────┘       └─────────────────────┘
```

## Key Insight

Agent fleets are probability distributions over state space. Optimal
transport tells you the cheapest way to reconfigure one fleet into
another. Spectral clustering groups similar agents before transport,
reducing computational cost. Monadic composition lets you chain transport
plans like state transitions.

## Example 1: Spectral Pre-Clustering Before Transport

Group agents by similarity, then compute transport between cluster
centroids rather than individual agents.

```rust
use spectral_fleet::kmeans::kmeans;
use wasserstein_agents::agents::AgentDistribution;
use wasserstein_agents::transport::OptimalTransport;
use rand::thread_rng;

fn clustered_transport(source: &AgentDistribution, target: &AgentDistribution) {
    let mut rng = thread_rng();

    // Cluster source agents
    let source_clusters = kmeans(&source.positions, 3, 50, &mut rng).unwrap();
    println!("Source cluster labels: {:?}", source_clusters.labels);

    // Build centroid-to-centroid cost matrix
    let mut cost = vec![vec![0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let d: f64 = source_clusters.centroids[i].iter()
                .zip(&target.positions[j % target.positions.len()])
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            cost[i][j] = d.sqrt();
        }
    }

    let source_weights = vec![1.0 / 3.0; 3];
    let target_weights = vec![1.0 / 3.0; 3];
    let w2 = OptimalTransport::wasserstein_2(&cost, &source_weights, &target_weights);
    println!("Clustered W2 distance: {:.4}", w2);
}
```

## Example 2: JKO Gradient Flow with Monadic State Tracking

Evolve a distribution toward a target using JKO flow, tracking the
intermediate states with the state monad.

```rust
use wasserstein_agents::gradient_flow::JKOScheme;
use wasserstein_agents::agents::AgentDistribution;
use categorical_agents::monad::StateMonad;

fn monadic_flow(initial: &AgentDistribution) -> Vec<AgentDistribution> {
    let jko = JKOScheme::new(0.1, 20, 0.01);

    // Gradient of quadratic potential: V(x) = |x|^2 / 2
    let grad_v = |x: &[f64]| x.iter().map(|&xi| xi).collect();

    let trajectory = jko.flow_with_potential(initial, grad_v);

    // Track total Wasserstein distance traveled via state monad
    let dist_state = StateMonad::new(|acc: f64| {
        let total: f64 = jko.wasserstein_trajectory(&trajectory).iter().sum();
        (total, acc + total)
    });

    let (total_dist, _) = dist_state.eval(0.0);
    println!("Total Wasserstein distance traveled: {:.4}", total_dist);
    trajectory
}
```

## Example 3: Optimal Assignment Between Fleets

Compute the optimal assignment matrix between two agent fleets and use it
to guide rebalancing.

```rust
use wasserstein_agents::agents::AgentDistribution;
use wasserstein_agents::transport::SinkhornSolver;

fn optimal_rebalancing_plan(fleet_a: &AgentDistribution, fleet_b: &AgentDistribution) {
    let cost = fleet_a.cross_cost_matrix(fleet_b);
    let n = fleet_a.len();
    let weights = vec![1.0 / n as f64; n];

    let solver = SinkhornSolver::new(0.01);
    let plan = solver.solve(&cost, &weights, &weights);

    let transport_cost = SinkhornSolver::transport_cost(&plan, &cost);
    println!("Optimal transport cost: {:.4}", transport_cost);

    // Plan[i][j] tells us how much of agent i's "mass" should move to agent j
    for i in 0..n.min(5) {
        for j in 0..n.min(5) {
            if plan[i][j] > 1e-3 {
                println!("Move {:.3} from agent {} to agent {}", plan[i][j], i, j);
            }
        }
    }
}
```

## Cargo.toml Wiring

```toml
[dependencies]
wasserstein-agents = { git = "https://github.com/SuperInstance/wasserstein-agents-rs" }
spectral-fleet = { git = "https://github.com/SuperInstance/spectral-fleet-rs" }
categorical-agents = { git = "https://github.com/SuperInstance/categorical-agents-rs" }
```

## Design Patterns

### Pattern: Multi-Scale Fleet Reconfiguration

Use JKO flow at coarse scale, then Sinkhorn at fine scale:

```rust
use wasserstein_agents::gradient_flow::JKOScheme;
use wasserstein_agents::transport::SinkhornSolver;
use wasserstein_agents::agents::AgentDistribution;

fn multiscale_reconfigure(fleet: &AgentDistribution, target: &AgentDistribution) {
    let jko = JKOScheme::new(0.2, 10, 0.05);
    let coarse = jko.flow_to_origin(fleet);
    let final_dist = coarse.last().unwrap();

    let cost = final_dist.cross_cost_matrix(target);
    let solver = SinkhornSolver::new(0.01);
    let plan = solver.solve(&cost, &fleet.weights, &target.weights);
    println!("Fine-scale transport cost: {:.4}",
        SinkhornSolver::transport_cost(&plan, &cost));
}
```
