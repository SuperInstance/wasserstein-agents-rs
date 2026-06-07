//! Agent distribution coordination via optimal transport.

use crate::transport::{OptimalTransport, SinkhornSolver};

/// An agent distribution: a probability distribution over agent states.
#[derive(Debug, Clone)]
pub struct AgentDistribution {
    /// Agent positions in state space (each inner vec is one agent's state).
    pub positions: Vec<Vec<f64>>,
    /// Probability mass at each position.
    pub weights: Vec<f64>,
}

impl AgentDistribution {
    /// Create a uniform distribution over positions.
    pub fn uniform(positions: Vec<Vec<f64>>) -> Self {
        let n = positions.len();
        let w = if n > 0 {
            vec![1.0 / n as f64; n]
        } else {
            vec![]
        };
        Self {
            positions,
            weights: w,
        }
    }

    /// Create from explicit weights.
    pub fn weighted(positions: Vec<Vec<f64>>, weights: Vec<f64>) -> Self {
        Self { positions, weights }
    }

    /// Number of agents.
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Dimension of the state space.
    pub fn dimension(&self) -> usize {
        self.positions.first().map(|p| p.len()).unwrap_or(0)
    }

    /// Mean position (center of mass).
    pub fn mean(&self) -> Vec<f64> {
        if self.is_empty() {
            return vec![];
        }
        let d = self.dimension();
        (0..d)
            .map(|k| {
                self.positions
                    .iter()
                    .zip(&self.weights)
                    .map(|(p, w)| p[k] * w)
                    .sum()
            })
            .collect()
    }

    /// Covariance matrix of the distribution.
    pub fn covariance(&self) -> Vec<Vec<f64>> {
        let d = self.dimension();
        if d == 0 {
            return vec![];
        }
        let mean = self.mean();
        let mut cov = vec![vec![0.0; d]; d];
        for (pos, w) in self.positions.iter().zip(&self.weights) {
            for i in 0..d {
                for j in 0..d {
                    cov[i][j] += w * (pos[i] - mean[i]) * (pos[j] - mean[j]);
                }
            }
        }
        cov
    }

    /// Compute pairwise Euclidean distance matrix.
    pub fn distance_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.len();
        (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        self.positions[i]
                            .iter()
                            .zip(&self.positions[j])
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>()
                            .sqrt()
                    })
                    .collect()
            })
            .collect()
    }

    /// Wasserstein distance to another agent distribution.
    pub fn wasserstein_distance(&self, other: &AgentDistribution) -> f64 {
        let cost = self.cross_cost_matrix(other);
        OptimalTransport::wasserstein_2(&cost, &self.weights, &other.weights)
    }

    /// Cross-cost matrix between self and other.
    pub fn cross_cost_matrix(&self, other: &AgentDistribution) -> Vec<Vec<f64>> {
        (0..self.len())
            .map(|i| {
                (0..other.len())
                    .map(|j| {
                        self.positions[i]
                            .iter()
                            .zip(&other.positions[j])
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>()
                    })
                    .collect()
            })
            .collect()
    }

    /// Find the optimal assignment of agents to target positions.
    /// Returns a transport plan matrix.
    pub fn optimal_assignment(&self, targets: &AgentDistribution) -> Vec<Vec<f64>> {
        let cost = self.cross_cost_matrix(targets);
        let solver = SinkhornSolver::new(0.01);
        solver.solve(&cost, &self.weights, &targets.weights)
    }

    /// Spread agents apart by moving them away from their centroid.
    pub fn spread(&self, factor: f64) -> Self {
        let mean = self.mean();
        let positions = self
            .positions
            .iter()
            .map(|p| {
                p.iter()
                    .zip(&mean)
                    .map(|(x, m)| m + (x - m) * factor)
                    .collect()
            })
            .collect();
        Self {
            positions,
            weights: self.weights.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform() {
        let dist = AgentDistribution::uniform(vec![vec![0.0], vec![1.0], vec![2.0]]);
        assert_eq!(dist.len(), 3);
        assert!((dist.weights[0] - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_mean() {
        let dist = AgentDistribution::uniform(vec![vec![0.0], vec![2.0]]);
        let m = dist.mean();
        assert!((m[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_covariance() {
        let dist = AgentDistribution::uniform(vec![vec![-1.0], vec![1.0]]);
        let cov = dist.covariance();
        assert!((cov[0][0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_distance_matrix() {
        let dist = AgentDistribution::uniform(vec![vec![0.0], vec![3.0], vec![6.0]]);
        let dm = dist.distance_matrix();
        assert!((dm[0][1] - 3.0).abs() < 1e-10);
        assert!((dm[0][2] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_spread() {
        let dist = AgentDistribution::uniform(vec![vec![0.0], vec![2.0]]);
        let spread = dist.spread(2.0);
        assert!((spread.positions[0][0] - (-1.0)).abs() < 1e-10);
        assert!((spread.positions[1][0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_wasserstein_same_distribution() {
        let dist = AgentDistribution::uniform(vec![vec![0.0], vec![1.0]]);
        let w = dist.wasserstein_distance(&dist);
        assert!(w < 0.1);
    }
}
