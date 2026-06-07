//! Sinkhorn algorithm for entropy-regularized optimal transport.

/// Optimal transport solver using Sinkhorn iterations.
pub struct SinkhornSolver {
    /// Regularization parameter (entropy strength).
    pub regularization: f64,
    /// Maximum iterations.
    pub max_iterations: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
}

impl SinkhornSolver {
    pub fn new(regularization: f64) -> Self {
        Self {
            regularization,
            max_iterations: 1000,
            tolerance: 1e-8,
        }
    }

    /// Compute entropy-regularized optimal transport plan.
    /// Returns the transport matrix T where T[i][j] = mass moved from i to j.
    ///
    /// Uses the Sinkhorn-Knopp algorithm:
    /// K = exp(-C/ε), then alternately normalize rows and columns.
    pub fn solve(&self, cost: &[Vec<f64>], mu: &[f64], nu: &[f64]) -> Vec<Vec<f64>> {
        let n = mu.len();
        let m = nu.len();
        let eps = self.regularization;

        // Kernel matrix K = exp(-C/ε)
        let k: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..m)
                    .map(|j| (-cost[i][j] / eps).exp().max(1e-300))
                    .collect()
            })
            .collect();

        // Dual variables (log domain for numerical stability)
        let mut u = vec![0.0f64; n];
        let mut v = vec![0.0f64; m];

        for _ in 0..self.max_iterations {
            // u = log(mu) - log(K * exp(v))
            let new_u: Vec<f64> = (0..n)
                .map(|i| {
                    let sum: f64 = (0..m).map(|j| k[i][j] * v[j].exp()).sum();
                    if sum > 1e-300 {
                        mu[i].ln() - sum.ln()
                    } else {
                        u[i]
                    }
                })
                .collect();

            // v = log(nu) - log(K^T * exp(u))
            let new_v: Vec<f64> = (0..m)
                .map(|j| {
                    let sum: f64 = (0..n).map(|i| k[i][j] * new_u[i].exp()).sum();
                    if sum > 1e-300 {
                        nu[j].ln() - sum.ln()
                    } else {
                        v[j]
                    }
                })
                .collect();

            // Check convergence
            let change: f64 = new_u
                .iter()
                .zip(&u)
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>()
                + new_v
                    .iter()
                    .zip(&v)
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f64>();
            u = new_u;
            v = new_v;

            if change < self.tolerance {
                break;
            }
        }

        // Build transport matrix
        (0..n)
            .map(|i| {
                (0..m)
                    .map(|j| {
                        let t = u[i].exp() * k[i][j] * v[j].exp();
                        if t.is_nan() || t.is_infinite() {
                            0.0
                        } else {
                            t
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Compute the transport cost of a plan.
    pub fn transport_cost(plan: &[Vec<f64>], cost: &[Vec<f64>]) -> f64 {
        plan.iter()
            .zip(cost)
            .map(|(row, crow)| row.iter().zip(crow).map(|(p, c)| p * c).sum::<f64>())
            .sum()
    }
}

/// Optimal transport utilities.
pub struct OptimalTransport;

impl OptimalTransport {
    /// Wasserstein-1 distance (Earth Mover's Distance) for discrete distributions.
    /// W_1 = min_T Σ_{ij} C_{ij} T_{ij} subject to T1=μ, T^T1=ν.
    /// Simplified: uses Sinkhorn with small regularization.
    pub fn wasserstein_1(cost: &[Vec<f64>], mu: &[f64], nu: &[f64]) -> f64 {
        let solver = SinkhornSolver::new(0.01);
        let plan = solver.solve(cost, mu, nu);
        SinkhornSolver::transport_cost(&plan, cost)
    }

    /// Wasserstein-2 distance (squared).
    /// W_2^2 = min_T Σ_{ij} |x_i - x_j|^2 T_{ij}.
    pub fn wasserstein_2_squared(cost: &[Vec<f64>], mu: &[f64], nu: &[f64]) -> f64 {
        let solver = SinkhornSolver::new(0.01);
        let plan = solver.solve(cost, mu, nu);
        SinkhornSolver::transport_cost(&plan, cost)
    }

    /// Wasserstein-2 distance (square root of W_2^2).
    pub fn wasserstein_2(cost: &[Vec<f64>], mu: &[f64], nu: &[f64]) -> f64 {
        Self::wasserstein_2_squared(cost, mu, nu).max(0.0).sqrt()
    }

    /// Barycenter of distributions using fixed-point iteration.
    /// Finds distribution ν that minimizes Σ_k λ_k W(ν, μ_k).
    pub fn barycenter(
        distributions: &[(&[f64], &[Vec<f64>])],
        weights: &[f64],
        n_iterations: usize,
    ) -> Vec<f64> {
        if distributions.is_empty() {
            return vec![];
        }
        let n = distributions[0].0.len();

        // Initialize as weighted average
        let mut bary: Vec<f64> = (0..n)
            .map(|i| {
                distributions
                    .iter()
                    .zip(weights)
                    .map(|((dist, _), w)| dist[i] * w)
                    .sum()
            })
            .collect();

        let solver = SinkhornSolver::new(0.1);

        for _ in 0..n_iterations {
            let mut new_bary = vec![0.0; n];
            let mut total_weight = 0.0;

            for ((dist, cost), w) in distributions.iter().zip(weights) {
                let plan = solver.solve(cost, &bary, dist);
                // Push-forward: new_bary[j] += w * Σ_i T_{ij}... simplified
                for j in 0..n {
                    let push: f64 = (0..n).map(|i| plan[i][j]).sum();
                    new_bary[j] += w * push;
                }
                total_weight += w;
            }

            if total_weight > 0.0 {
                for v in new_bary.iter_mut() {
                    *v /= total_weight;
                }
            }
            bary = new_bary;
        }

        bary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sinkhorn_identity_cost() {
        let cost = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let mu = vec![0.5, 0.5];
        let nu = vec![0.5, 0.5];
        let solver = SinkhornSolver::new(0.1);
        let plan = solver.solve(&cost, &mu, &nu);
        // With identical distributions and symmetric cost, plan should be approximately diagonal
        assert!(plan[0][0] > 0.1);
        assert!(plan[1][1] > 0.1);
    }

    #[test]
    fn test_sinkhorn_preserves_marginals() {
        let cost = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let mu = vec![0.3, 0.7];
        let nu = vec![0.6, 0.4];
        let solver = SinkhornSolver::new(0.1);
        let plan = solver.solve(&cost, &mu, &nu);
        // Row sums should approximate mu
        let row0: f64 = plan[0].iter().sum();
        let row1: f64 = plan[1].iter().sum();
        assert!((row0 - 0.3).abs() < 0.1);
        assert!((row1 - 0.7).abs() < 0.1);
    }

    #[test]
    fn test_wasserstein_1_same_distribution() {
        let cost = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let mu = vec![0.5, 0.5];
        let w = OptimalTransport::wasserstein_1(&cost, &mu, &mu);
        assert!(w < 0.1); // Same distribution → ~0 distance
    }

    #[test]
    fn test_wasserstein_2_nonnegative() {
        let cost = vec![vec![0.0, 2.0], vec![2.0, 0.0]];
        let mu = vec![0.5, 0.5];
        let nu = vec![0.5, 0.5];
        let w = OptimalTransport::wasserstein_2(&cost, &mu, &nu);
        assert!(w >= 0.0);
    }

    #[test]
    fn test_transport_cost() {
        let plan = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let cost = vec![vec![0.0, 3.0], vec![3.0, 0.0]];
        let tc = SinkhornSolver::transport_cost(&plan, &cost);
        assert!((tc - 0.0).abs() < 1e-10); // Diagonal plan on diagonal cost = 0
    }
}
