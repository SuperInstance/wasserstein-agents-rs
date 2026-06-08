# wasserstein-agents-rs

Optimal transport between agent distributions.

Moving agents from where they are to where they need to be — with minimum cost. This crate computes transport plans, Wasserstein distances, barycenters, and gradient flows for multi-agent fleet coordination.

The Wasserstein distance measures the "earth mover's cost" of reshaping one distribution into another. For agents, it answers: how much total effort does it take to redistribute a fleet from configuration $\mu$ to configuration $\nu$?

Part of the **sunset-ecosystem**: `dial-theory-rs` provides tradition positions that define agent distributions, `conservation-law` enforces mass conservation during transport, and `si-fleet-api` executes the resulting transport plans across the fleet.

## The Math

### Optimal Transport

Given source distribution $\mu \in \mathbb{R}^n$ and target $\nu \in \mathbb{R}^m$ with cost matrix $C \in \mathbb{R}^{n \times m}$, the optimal transport problem is:

$$\min_{T \geq 0} \sum_{i,j} C_{ij} T_{ij} \quad \text{subject to} \quad T\mathbf{1} = \mu, \quad T^\top\mathbf{1} = \nu$$

### Sinkhorn Algorithm

Entropy-regularized transport replaces the objective with:

$$\min_T \sum_{i,j} C_{ij} T_{ij} + \varepsilon \sum_{i,j} T_{ij}(\log T_{ij} - 1)$$

The Sinkhorn-Knopp algorithm solves this by iteratively normalizing rows and columns of the kernel $K = \exp(-C/\varepsilon)$:

$$u \leftarrow \frac{\mu}{Kv}, \quad v \leftarrow \frac{\nu}{K^\top u}$$

The transport plan is $T = \text{diag}(u) K \text{diag}(v)$.

### Wasserstein Distances

- **Wasserstein-1**: $W_1(\mu, \nu) = \min_T \sum_{i,j} C_{ij} T_{ij}$
- **Wasserstein-2**: $W_2^2(\mu, \nu) = \min_T \sum_{i,j} |x_i - x_j|^2 T_{ij}$

### Wasserstein Barycenter

The barycenter of $K$ distributions $\{\mu_1, \ldots, \mu_K\}$ is:

$$\nu^* = \text{argmin}_\nu \frac{1}{K} \sum_{k=1}^K W_p(\nu, \mu_k)$$

For 1D distributions, this is simply the average of quantile functions. For multi-dimensional distributions, we use iterative Bregman projection.

### JKO Gradient Flow

The Jordan-Kinderlehrer-Otto scheme evolves a distribution $\mu_t$ by:

$$\mu_{t+1} = \text{argmin}_\nu \left\{ \tau \mathcal{F}(\nu) + \frac{1}{2} W_2^2(\mu_t, \nu) \right\}$$

where $\mathcal{F}$ is a free energy functional and $\tau$ is the time step.

### Sliced Wasserstein Distance

Projects high-dimensional distributions onto random 1D lines sampled from the unit sphere $S^{d-1}$:

$$SW_p(\mu, \nu) = \mathbb{E}_{\theta \sim \mathcal{U}(S^{d-1})} \left[ W_p(P_\theta \mu, P_\theta \nu) \right]$$

Estimated with $L$ random projections.

## Installation

```toml
[dependencies]
wasserstein-agents-rs = { git = "https://github.com/SuperInstance/wasserstein-agents-rs" }
```

## Usage

### Sinkhorn Optimal Transport

```rust
use wasserstein_agents_rs::transport::{SinkhornSolver, OptimalTransport};

// Cost matrix: moving agent i to position j costs C[i][j]
let cost = vec![
    vec![0.0, 1.0, 2.0],
    vec![1.0, 0.0, 1.0],
    vec![2.0, 1.0, 0.0],
];

// Source and target distributions
let mu = vec![0.5, 0.3, 0.2]; // current agent positions
let nu = vec![0.2, 0.5, 0.3]; // target positions

// Solve with Sinkhorn (regularization ε = 0.1)
let solver = SinkhornSolver::new(0.1);
let plan = solver.solve(&cost, &mu, &nu);

println!("Transport plan:");
for row in &plan {
    println!("  {:?}", row);
}

// Total transport cost
let total_cost = SinkhornSolver::transport_cost(&plan, &cost);
println!("Total cost: {:.4}", total_cost);
```

### Wasserstein Distances

```rust
use wasserstein_agents_rs::transport::OptimalTransport;

let cost = vec![
    vec![0.0, 1.0, 4.0],
    vec![1.0, 0.0, 1.0],
    vec![4.0, 1.0, 0.0],
];

let mu = vec![0.5, 0.3, 0.2];
let nu = vec![0.2, 0.3, 0.5];

// W₁: Earth Mover's Distance
let w1 = OptimalTransport::wasserstein_1(&cost, &mu, &nu);
println!("W₁ = {:.4}", w1);

// W₂²: squared Wasserstein-2
let w2_sq = OptimalTransport::wasserstein_2_squared(&cost, &mu, &nu);
println!("W₂² = {:.4}", w2_sq);

// W₂: Wasserstein-2 distance
let w2 = OptimalTransport::wasserstein_2(&cost, &mu, &nu);
println!("W₂ = {:.4}", w2);

// Same distribution → zero distance
let w_same = OptimalTransport::wasserstein_1(&cost, &mu, &mu);
println!("Self-distance: {:.4}", w_same); // ≈ 0
```

### Agent Distributions

```rust
use wasserstein_agents_rs::agents::AgentDistribution;

// Create a fleet distribution: 3 agents in 2D space
let fleet = AgentDistribution::uniform(vec![
    vec![0.0, 0.0],
    vec![1.0, 0.0],
    vec![0.0, 1.0],
]);

println!("Fleet size: {}", fleet.len());
println!("State dimension: {}", fleet.dimension());

// Compute fleet statistics
let mean = fleet.mean();
println!("Mean position: {:?}", mean);

let cov = fleet.covariance();
println!("Covariance: {:?}", cov);

// Distance to another configuration
let target = AgentDistribution::uniform(vec![
    vec![1.0, 1.0],
    vec![2.0, 1.0],
    vec![1.0, 2.0],
]);
let w_dist = fleet.wasserstein_distance(&target);
println!("Wasserstein distance to target: {:.4}", w_dist);

// Find optimal assignment plan
let plan = fleet.optimal_assignment(&target);
println!("Assignment plan:");
for row in &plan {
    println!("  {:?}", row);
}

// Spread agents apart from centroid
let spread = fleet.spread(2.0);
println!("Spread positions: {:?}", spread.positions);
```

### Wasserstein Barycenter

```rust
use wasserstein_agents_rs::barycenter::{
    barycenter_1d_quantile, barycenter_sinkhorn, barycenter_free_support, dist_w2,
};

// 1D barycenter: average of quantile functions
let dist_a = vec![0.0, 1.0, 2.0, 3.0];
let dist_b = vec![2.0, 3.0, 4.0, 5.0];
let dist_c = vec![1.0, 2.0, 3.0, 4.0];

let bary_1d = barycenter_1d_quantile(&[dist_a, dist_b, dist_c]);
println!("1D barycenter: {:?}", bary_1d); // [1.0, 2.0, 3.0, 4.0]

// Free-support barycenter: iterative displacement interpolation
let cloud_a = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
let cloud_b = vec![vec![0.0, 2.0], vec![1.0, 2.0]];
let cloud_c = vec![vec![0.0, 1.0], vec![2.0, 1.0]];

let bary_free = barycenter_free_support(&[cloud_a, cloud_b, cloud_c], 20, 1e-6);
println!("Free-support barycenter:");
for pt in &bary_free {
    println!("  {:?}", pt);
}

// Distance between point clouds
let d = dist_w2(&bary_free, &cloud_a);
println!("W₂ from barycenter to cloud_a: {:.4}", d);
```

### Sinkhorn Barycenter on a Grid

```rust
use wasserstein_agents_rs::barycenter::barycenter_sinkhorn;

// Three distributions on a 5-point grid
let dist_1 = vec![0.5, 0.3, 0.1, 0.05, 0.05];
let dist_2 = vec![0.05, 0.05, 0.1, 0.3, 0.5];
let dist_3 = vec![0.1, 0.2, 0.4, 0.2, 0.1];

// Squared-distance cost matrix for the grid
let n = 5;
let cost: Vec<Vec<f64>> = (0..n)
    .map(|i| {
        (0..n)
            .map(|j| ((i as f64 - j as f64).powi(2)))
            .collect()
    })
    .collect();

// Compute barycenter via Sinkhorn iterations
let bary = barycenter_sinkhorn(&[dist_1, dist_2, dist_3], &cost, 0.1, 100, 1e-8);
println!("Sinkhorn barycenter: {:?}", bary);
```

### JKO Gradient Flow

```rust
use wasserstein_agents_rs::agents::AgentDistribution;
use wasserstein_agents_rs::gradient_flow::JKOScheme;

// Start with agents spread out
let initial = AgentDistribution::uniform(vec![
    vec![3.0],
    vec![-3.0],
    vec![1.5],
]);

// Run JKO with quadratic potential V(x) = 0.5*|x|²
// This drives agents toward the origin
let jko = JKOScheme::new(0.1, 50);
let trajectory = jko.flow_to_origin(&initial);

println!("JKO trajectory ({} steps):", trajectory.len());
for (t, dist) in trajectory.iter().enumerate() {
    let mean = dist.mean();
    if t % 10 == 0 || t == trajectory.len() - 1 {
        println!("  t={:3}: mean={:.4}", t, mean[0]);
    }
}

// Wasserstein distance between consecutive steps
let w_traj = jko.wasserstein_trajectory(&trajectory);
println!("\nStep-wise Wasserstein distances:");
for (i, w) in w_traj.iter().enumerate() {
    if i < 5 || i >= w_traj.len() - 2 {
        println!("  step {}→{}: W₂ = {:.6}", i, i+1, w);
    }
}
```

### JKO with Custom Potential

```rust
use wasserstein_agents_rs::agents::AgentDistribution;
use wasserstein_agents_rs::gradient_flow::JKOScheme;

let initial = AgentDistribution::uniform(vec![
    vec![2.0, 1.0],
    vec![-1.0, 2.0],
    vec![0.0, -2.0],
]);

// Custom potential: V(x) = 0.5 * (x₁² + x₂²) → ∇V = x
let grad_v = |x: &[f64]| x.to_vec();

let jko = JKOScheme::new(0.05, 100);
let trajectory = jko.flow_with_potential(&initial, grad_v);

let final_dist = trajectory.last().unwrap();
let mean = final_dist.mean();
println!("Final mean position: {:?}", mean); // near origin
```

### Sliced Wasserstein Distance

```rust
use wasserstein_agents_rs::sliced::{sliced_wasserstein_1, sliced_wasserstein_2, sliced_wasserstein_custom};

// Two point clouds in 2D
let cloud_a = vec![
    vec![0.0, 0.0],
    vec![1.0, 0.0],
    vec![0.0, 1.0],
    vec![1.0, 1.0],
];

let cloud_b = vec![
    vec![1.0, 1.0],
    vec![2.0, 1.0],
    vec![1.0, 2.0],
    vec![2.0, 2.0],
];

// Sliced W₁ with 100 random projections
let sw1 = sliced_wasserstein_1(&cloud_a, &cloud_b, 100, 42);
println!("Sliced W₁ (100 projections): {:.4}", sw1);

// Sliced W₂ with 1000 projections (more accurate)
let sw2 = sliced_wasserstein_2(&cloud_a, &cloud_b, 1000, 42);
println!("Sliced W₂ (1000 projections): {:.4}", sw2);

// Same distribution → zero distance
let sw_self = sliced_wasserstein_1(&cloud_a, &cloud_a, 100, 42);
println!("Self-distance: {:.6}", sw_self); // ≈ 0

// Custom projection directions
let dirs = vec![
    vec![1.0, 0.0],  // x-axis
    vec![0.0, 1.0],  // y-axis
    vec![0.707, 0.707], // diagonal
];
let sw_custom = sliced_wasserstein_custom(&cloud_a, &cloud_b, &dirs);
println!("Custom sliced W₁: {:.4}", sw_custom);
```

### Pairwise Distance Matrix

```rust
use wasserstein_agents_rs::agents::AgentDistribution;

let fleet = AgentDistribution::uniform(vec![
    vec![0.0, 0.0],
    vec![3.0, 4.0],
    vec![6.0, 8.0],
]);

let dm = fleet.distance_matrix();
println!("Pairwise distance matrix:");
for row in &dm {
    println!("  {:?}", row);
}
// [0, 5, 10]
// [5, 0, 5]
// [10, 5, 0]
```

## API Reference

### Transport

| Function | Description |
|----------|-------------|
| `SinkhornSolver::new(ε)` | Create solver with regularization $\varepsilon$ |
| `solver.solve(cost, μ, ν)` | Compute optimal transport plan |
| `SinkhornSolver::transport_cost(plan, cost)` | Total cost of a plan |
| `OptimalTransport::wasserstein_1(cost, μ, ν)` | $W_1$ distance |
| `OptimalTransport::wasserstein_2(cost, μ, ν)` | $W_2$ distance |
| `OptimalTransport::wasserstein_2_squared(cost, μ, ν)` | $W_2^2$ distance |

### Agent Distributions

| Method | Description |
|--------|-------------|
| `AgentDistribution::uniform(positions)` | Uniform weights |
| `AgentDistribution::weighted(positions, weights)` | Custom weights |
| `mean()` | Center of mass |
| `covariance()` | Covariance matrix |
| `distance_matrix()` | Pairwise Euclidean distances |
| `wasserstein_distance(other)` | $W_2$ to another distribution |
| `optimal_assignment(targets)` | Transport plan to targets |
| `spread(factor)` | Scale positions from centroid |

### Barycenter

| Function | Description |
|----------|-------------|
| `barycenter_1d_quantile(distributions)` | Average quantile functions |
| `barycenter_sinkhorn(dists, cost, ε, iter, tol)` | Sinkhorn barycenter on grid |
| `barycenter_free_support(dists, iter, tol)` | Free-support via displacement |

### Gradient Flow

| Method | Description |
|--------|-------------|
| `JKOScheme::new(τ, steps)` | Create JKO with timestep $\tau$ |
| `flow_to_origin(initial)` | Flow under quadratic potential |
| `flow_with_potential(initial, ∇V)` | Flow under custom potential |
| `wasserstein_trajectory(traj)` | $W_2$ between consecutive steps |

### Sliced Wasserstein

| Function | Description |
|----------|-------------|
| `sliced_wasserstein_1(a, b, L, seed)` | Sliced $W_1$ with $L$ projections |
| `sliced_wasserstein_2(a, b, L, seed)` | Sliced $W_2$ with $L$ projections |
| `sliced_wasserstein_custom(a, b, dirs)` | Sliced with custom directions |

## Why This Matters for Agent Systems

1. **Fleet rebalancing**: When agents cluster too tightly, optimal transport computes the minimum-cost redistribution plan.
2. **Distributional drift**: Wasserstein distance detects when a fleet's state distribution has shifted from its target — more robust than comparing means.
3. **Barycenter aggregation**: When multiple fleets merge, the Wasserstein barycenter finds the consensus distribution that minimizes total transport cost.
4. **Gradient flow optimization**: JKO flow provides a principled way to evolve agent distributions toward objectives while respecting geometry.
5. **Scalable comparison**: Sliced Wasserstein gives fast $O(n \log n)$ approximations of the transport distance in high dimensions.

## Integration

### With `dial-theory-rs`

```rust
// dial-theory-rs provides tradition positions
// wasserstein-agents-rs measures distributional distance between agent clusters
// Transport plans move agents between cultural configurations
```

### With `conservation-law`

```rust
// Transport plans must conserve total mass: Σ T_{ij} = 1
// conservation-law verifies this invariant during plan execution
```

### With `si-fleet-api`

```rust
// Computed transport plans are dispatched via si-fleet-api
// Each agent receives its assignment and moves toward its target
```

## License

MIT
