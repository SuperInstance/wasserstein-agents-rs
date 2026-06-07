//! Wasserstein gradient flow via the JKO (Jordan-Kinderlehrer-Otto) scheme.

use crate::agents::AgentDistribution;
use crate::transport::SinkhornSolver;

/// JKO scheme for Wasserstein gradient flow.
///
/// The JKO scheme evolves a distribution μ_t by:
/// μ_{t+1} = argmin_ν { τ * F(ν) + W_2^2(μ_t, ν) / 2 }
///
/// where F is a free energy functional and τ is the time step.
pub struct JKOScheme {
    /// Time step.
    pub dt: f64,
    /// Number of JKO steps to perform.
    pub n_steps: usize,
    /// Entropy regularization for Sinkhorn.
    pub regularization: f64,
}

impl JKOScheme {
    pub fn new(dt: f64, n_steps: usize) -> Self {
        Self {
            dt,
            n_steps,
            regularization: 0.1,
        }
    }

    /// Run JKO with a quadratic potential V(x) = 0.5 * |x|².
    /// This drives the distribution toward the origin (heat equation).
    pub fn flow_to_origin(&self, initial: &AgentDistribution) -> Vec<AgentDistribution> {
        let mut trajectory = vec![initial.clone()];
        let mut current = initial.clone();

        for _ in 0..self.n_steps {
            current = self.jko_step_quadratic(&current);
            trajectory.push(current.clone());
        }

        trajectory
    }

    /// One JKO step with quadratic potential.
    fn jko_step_quadratic(&self, dist: &AgentDistribution) -> AgentDistribution {
        let n = dist.len();
        let _d = dist.dimension();
        if n == 0 {
            return dist.clone();
        }

        let solver = SinkhornSolver::new(self.regularization);

        // The JKO update for quadratic potential is:
        // Move each particle toward the origin by dt
        // (This is the proximal operator of the quadratic potential)
        let positions = dist
            .positions
            .iter()
            .map(|p| p.iter().map(|x| x / (1.0 + self.dt)).collect())
            .collect();

        // Compute cost between current and new positions
        let new_dist = AgentDistribution {
            positions,
            weights: dist.weights.clone(),
        };

        let cost = dist.cross_cost_matrix(&new_dist);
        let plan = solver.solve(&cost, &dist.weights, &new_dist.weights);

        // Update weights based on transport plan
        let mut new_weights = vec![0.0; n];
        for j in 0..n {
            for i in 0..n {
                new_weights[j] += plan[i][j];
            }
        }

        // Normalize
        let total: f64 = new_weights.iter().sum();
        if total > 0.0 {
            for w in new_weights.iter_mut() {
                *w /= total;
            }
        }

        AgentDistribution {
            positions: new_dist.positions,
            weights: new_weights,
        }
    }

    /// Run JKO with a custom potential gradient.
    /// grad_V maps position to gradient of the potential.
    pub fn flow_with_potential<F>(
        &self,
        initial: &AgentDistribution,
        grad_v: F,
    ) -> Vec<AgentDistribution>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let mut trajectory = vec![initial.clone()];
        let mut current = initial.clone();

        for _ in 0..self.n_steps {
            let positions = current
                .positions
                .iter()
                .map(|p| {
                    let g = grad_v(p);
                    p.iter().zip(&g).map(|(x, gv)| x - self.dt * gv).collect()
                })
                .collect();
            current = AgentDistribution {
                positions,
                weights: current.weights.clone(),
            };
            trajectory.push(current.clone());
        }

        trajectory
    }

    /// Compute the 2-Wasserstein distance between consecutive distributions.
    pub fn wasserstein_trajectory(&self, trajectory: &[AgentDistribution]) -> Vec<f64> {
        trajectory
            .windows(2)
            .map(|w| w[0].wasserstein_distance(&w[1]))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jko_converges_to_origin() {
        let initial = AgentDistribution::uniform(vec![vec![2.0], vec![-2.0]]);
        let jko = JKOScheme::new(0.1, 50);
        let traj = jko.flow_to_origin(&initial);
        let final_pos = &traj.last().unwrap().positions;
        // Should be close to origin
        assert!(final_pos[0][0].abs() < 1.0);
        assert!(final_pos[1][0].abs() < 1.0);
    }

    #[test]
    fn test_jko_trajectory_length() {
        let initial = AgentDistribution::uniform(vec![vec![1.0], vec![-1.0]]);
        let jko = JKOScheme::new(0.1, 5);
        let traj = jko.flow_to_origin(&initial);
        assert_eq!(traj.len(), 6); // initial + 5 steps
    }

    #[test]
    fn test_jko_custom_potential() {
        let initial = AgentDistribution::uniform(vec![vec![1.0], vec![-1.0]]);
        let jko = JKOScheme::new(0.1, 10);
        let grad_v = |x: &[f64]| x.to_vec(); // V(x) = 0.5*|x|² → ∇V = x
        let traj = jko.flow_with_potential(&initial, grad_v);
        assert_eq!(traj.len(), 11);
        // Should move toward origin
        let final_pos = &traj.last().unwrap().positions;
        assert!(final_pos[0][0].abs() < 0.5);
    }

    #[test]
    fn test_jko_preserves_total_mass() {
        let initial = AgentDistribution::uniform(vec![vec![1.0], vec![2.0], vec![3.0]]);
        let jko = JKOScheme::new(0.1, 10);
        let traj = jko.flow_to_origin(&initial);
        let final_dist = traj.last().unwrap();
        let total: f64 = final_dist.weights.iter().sum();
        assert!((total - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_wasserstein_trajectory() {
        let initial = AgentDistribution::uniform(vec![vec![1.0], vec![-1.0]]);
        let jko = JKOScheme::new(0.1, 3);
        let traj = jko.flow_to_origin(&initial);
        let w_traj = jko.wasserstein_trajectory(&traj);
        assert_eq!(w_traj.len(), 3);
    }
}
