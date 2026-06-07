# wasserstein-agents

> Wasserstein distance, optimal transport, and agent distribution coordination in Rust.

## What This Does

This crate computes entropy-regularized optimal transport plans using the Sinkhorn-Knopp algorithm, measures Wasserstein-1 and Wasserstein-2 distances between discrete distributions, computes Wasserstein barycenters for fusing multiple agent populations, and evolves distributions over time via the JKO (Jordan-Kinderlehrer-Otto) gradient flow scheme. The "agents" framing treats every probability distribution as a fleet of particles in state space, and optimal transport as the coordination protocol that reshapes one fleet into another with minimal effort.

## Why It Matters

Most machine learning compares distributions with KL divergence or total variation — metrics that ignore geometry. The Wasserstein distance respects the underlying space: moving an agent one meter costs less than moving it a kilometer. In the AGI trajectory, this is the difference between coordination protocols that are blind to physical reality and those that are grounded in it. When your fleet's planning layer reasons in Wasserstein space, it naturally prefers smooth, efficient, and physically plausible reconfigurations.

## Quick Start

```bash
cargo add wasserstein-agents
```

```rust
use wasserstein_agents::agents::AgentDistribution;
use wasserstein_agents::transport::OptimalTransport;
use wasserstein_agents::gradient_flow::JKOScheme;

fn main() {
    // Two agent fleets in 1D state space
    let fleet_a = AgentDistribution::uniform(vec![vec![0.0], vec![2.0], vec![4.0]]);
    let fleet_b = AgentDistribution::uniform(vec![vec![1.0], vec![3.0], vec![5.0]]);

    // Wasserstein-2 distance between fleets
    let w2 = fleet_a.wasserstein_distance(&fleet_b);
    println!("W₂ distance: {:.4}", w2);

    // Optimal assignment plan
    let plan = fleet_a.optimal_assignment(&fleet_b);
    println!("Transport plan shape: {}×{}", plan.len(), plan[0].len());

    // Evolve fleet toward origin via JKO gradient flow
    let jko = JKOScheme::new(0.1, 20);
    let trajectory = jko.flow_to_origin(&fleet_a);
    let final_dist = trajectory.last().unwrap();
    println!("Final mean: {:.4?}", final_dist.mean());
}
```

## Architecture

| Module | Purpose |
|--------|---------|
| `transport` | Sinkhorn-Knopp solver, W₁/W₂ distances, barycenter fixed-point iteration |
| `agents` | `AgentDistribution` with mean, covariance, distance matrix, and optimal assignment |
| `gradient_flow` | JKO scheme for Wasserstein gradient flow with quadratic and custom potentials |

## API Tour

### `SinkhornSolver`

Entropy-regularized optimal transport via iterative row/column normalization in the log domain.

```rust
impl SinkhornSolver {
    pub fn new(regularization: f64) -> Self;
    pub fn solve(&self, cost: &[Vec<f64>], mu: &[f64], nu: &[f64]) -> Vec<Vec<f64>>;
    pub fn transport_cost(plan: &[Vec<f64>], cost: &[Vec<f64>]) -> f64;
}
```

```rust
let solver = SinkhornSolver::new(0.1);
let plan = solver.solve(&cost, &source_weights, &target_weights);
```

### `OptimalTransport`

Static utilities for Wasserstein distances and barycenters.

```rust
impl OptimalTransport {
    pub fn wasserstein_1(cost: &[Vec<f64>], mu: &[f64], nu: &[f64]) -> f64;
    pub fn wasserstein_2(cost: &[Vec<f64>], mu: &[f64], nu: &[f64]) -> f64;
    pub fn wasserstein_2_squared(cost: &[Vec<f64>], mu: &[f64], nu: &[f64]) -> f64;
    pub fn barycenter(
        distributions: &[(&[f64], &[Vec<f64>])],
        weights: &[f64],
        n_iterations: usize,
    ) -> Vec<f64>;
}
```

### `AgentDistribution`

A probability distribution over agent state vectors.

```rust
impl AgentDistribution {
    pub fn uniform(positions: Vec<Vec<f64>>) -> Self;
    pub fn weighted(positions: Vec<Vec<f64>>, weights: Vec<f64>) -> Self;
    pub fn mean(&self) -> Vec<f64>;
    pub fn covariance(&self) -> Vec<Vec<f64>>;
    pub fn distance_matrix(&self) -> Vec<Vec<f64>>;
    pub fn wasserstein_distance(&self, other: &AgentDistribution) -> f64;
    pub fn optimal_assignment(&self, targets: &AgentDistribution) -> Vec<Vec<f64>>;
    pub fn spread(&self, factor: f64) -> Self;
}
```

### `JKOScheme`

Wasserstein gradient flow via proximal steps.

```rust
impl JKOScheme {
    pub fn new(dt: f64, n_steps: usize) -> Self;
    pub fn flow_to_origin(&self, initial: &AgentDistribution) -> Vec<AgentDistribution>;
    pub fn flow_with_potential<F>(&self, initial: &AgentDistribution, grad_v: F) -> Vec<AgentDistribution>
    where F: Fn(&[f64]) -> Vec<f64>;
    pub fn wasserstein_trajectory(&self, trajectory: &[AgentDistribution]) -> Vec<f64>;
}
```

## Performance

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Sinkhorn iteration | O(n × m) per iteration | n, m = distribution sizes |
| Wasserstein distance | O(iter × n × m) | Dominated by Sinkhorn convergence |
| Barycenter | O(iter × k × n²) | k = number of distributions |
| JKO step | O(n²) | One Sinkhorn solve + position update |
| Covariance | O(d² × n) | d = state space dimension |

The log-domain Sinkhorn implementation avoids underflow for small regularization values. For large-scale problems, consider multiscale or Nystrom approximations.

## Ecosystem

- **[conservation-law](https://github.com/SuperInstance/conservation-law-rs)** — Compare agent trajectories before and after transport via energy invariants
- **[spectral-fleet](https://github.com/SuperInstance/spectral-fleet-rs)** — Cluster agents before computing inter-cluster transport costs
- **[ga-core](https://github.com/SuperInstance/ga-core-rs)** — Embed agents in conformal space so transport respects geometric distances
- **[categorical-agents](https://github.com/SuperInstance/categorical-agents-rs)** — Compose transport plans monadically for multi-stage fleet redeployment

## Ideas for Improvement

1. **Sliced Wasserstein** — Reduce high-dimensional transport to 1D projections for O(n log n) approximate distances.
2. **GPU Sinkhorn** — Batch multiple transport problems on CUDA for fleet-scale coordination.
3. **Neural optimal transport** — Learn a potential network that maps source to target distributions directly.
4. **Time-dependent JKO** — Add drift and diffusion terms for Fokker-Planck-style agent motion.
5. **Multi-marginal transport** — Extend barycenter to true multi-marginal OT for fleet-wide consensus.

## License

MIT OR Apache-2.0
