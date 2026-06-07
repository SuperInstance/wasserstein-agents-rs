# Integration Guide: wasserstein-agents

## What This Crate Provides

- **`AgentDistribution`** — Probability distribution over agent states with positions and weights; supports uniform/weighted creation, mean, Wasserstein distance
- **`SinkhornSolver`** — Entropy-regularized optimal transport via Sinkhorn-Knopp iterations; produces transport matrices T[i][j]
- **`OptimalTransport`** — Transport plan computation between distributions
- **`JKOScheme`** — Wasserstein gradient flow via Jordan-Kinderlehrer-Otto scheme; evolves distributions toward minima
- **`barycenter_1d_quantile`**, **`barycenter_sinkhorn`**, **`barycenter_free_support`** — Wasserstein barycenters (Fréchet means in Wasserstein space)
- **`sliced_wasserstein_1`**, **`sliced_wasserstein_2`**, **`sliced_wasserstein_custom`** — Sliced Wasserstein distances for fast approximation

This crate provides optimal transport, Wasserstein distances, and distribution coordination for multi-agent fleets. It measures how different two agent distributions are (not just pointwise, but geometrically) and computes the optimal way to transform one into another.

## How to Add This Crate

```bash
cargo add wasserstein-agents
```

```rust
use wasserstein_agents::AgentDistribution;
use wasserstein_agents::SinkhornSolver;

let source = AgentDistribution::uniform(vec![vec![0.0, 0.0], vec![1.0, 0.0]]);
let target = AgentDistribution::uniform(vec![vec![1.0, 0.0], vec![2.0, 0.0]]);

let solver = SinkhornSolver::new(0.1); // regularization ε
let cost = source.cost_matrix(&target);
let plan = solver.solve(&cost, &source.weights, &target.weights);
println!("Transport plan: {:?}", plan);
```

## Integration Points

### categorical-agents

- **Why**: categorical-agents provides the composition algebra; wasserstein-agents provides the metric for measuring how far a composed agent distribution is from its target. Monadic composition produces intermediate distributions; Wasserstein distance quantifies progress.
- **How**: After composing agent behaviors categorically, use `SinkhornSolver` to measure the Wasserstein distance between the composed distribution and the desired target distribution.

```rust
use wasserstein_agents::AgentDistribution;
use wasserstein_agents::barycenter::dist_w2;

// Measure improvement after categorical composition
let before = AgentDistribution::uniform(vec![vec![0.0], vec![1.0], vec![2.0]]);
let after = AgentDistribution::weighted(
    vec![vec![0.2], vec![0.8], vec![1.8]],
    vec![0.33, 0.33, 0.34],
);

let improvement = dist_w2(&before, &after);
println!("Distribution shifted by W₂ = {:.4}", improvement);
```

### persistent-sheaf

- **Why**: persistent-sheaf provides topological data analysis via persistent sheaf cohomology; wasserstein-agents provides the geometric transport. Together: topology reveals structure, transport optimizes movement through that structure.
- **How**: Use persistent-sheaf's `SimplicialComplex` to discover topological structure in agent distributions, then use wasserstein-agents' `SinkhornSolver` to compute optimal transport plans that respect the discovered topology.

```rust
use wasserstein_agents::AgentDistribution;
use wasserstein_agents::SinkhornSolver;

// Agent positions → sheaf stalks → transport respects topology
let fleet = AgentDistribution::uniform(vec![
    vec![0.0, 0.0], vec![1.0, 0.0], vec![0.5, 0.866],
]);

let target = AgentDistribution::uniform(vec![
    vec![0.5, 0.0], vec![1.5, 0.0], vec![1.0, 0.866],
]);

let solver = SinkhornSolver::new(0.05);
let cost = fleet.cost_matrix(&target);
let plan = solver.solve(&cost, &fleet.weights, &target.weights);
```

## For AI Agents

- **Context needed**: Source and target agent positions, cost function (usually Euclidean distance), regularization parameter ε
- **Key imports**: `wasserstein_agents::{AgentDistribution, SinkhornSolver, JKOScheme}`
- **Integration pattern**: Create `AgentDistribution` → compute cost matrix → run `SinkhornSolver::solve()` → extract transport plan → measure improvement with W₂ distance
- **Error handling**: Sinkhorn iterations may not converge if ε is too small — increase regularization or max_iterations. Empty distributions return empty transport plans.

## For Humans

- **Prerequisites**: Basic optimal transport theory, Wasserstein distance concept, understanding of entropy regularization
- **Learning path**: Start with `transport.rs` (Sinkhorn algorithm), then `agents.rs` (agent distributions), then `gradient_flow.rs` (JKO scheme), then `barycenter.rs` (multi-distribution means)
- **Common pitfalls**:
  - Regularization ε too small → numerical instability (division by near-zero); too large → blurry transport plans
  - `AgentDistribution::uniform()` normalizes weights to sum to 1.0 — don't pass pre-normalized weights
  - JKO `flow_to_origin()` uses a fixed quadratic potential — for custom potentials, extend the `JKOScheme`
  - Sliced Wasserstein is an approximation — use full Sinkhorn for exact distances when n < 100
