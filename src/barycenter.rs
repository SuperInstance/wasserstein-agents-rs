//! Wasserstein barycenter — free-support barycenter via iterative Bregman projection.
//!
//! The Wasserstein barycenter of K distributions {μ_1, …, μ_K} is the
//! distribution ν that minimizes the sum of Wasserstein distances:
//!
//! ```text
//! ν* = argmin_ν  (1/K) Σ_k W_p(ν, μ_k)
//! ```
//!
//! This module implements a fixed-support barycenter using iterative
//! Bregman projections (Benamou et al., 2015), suitable for discrete
//! distributions on a common grid.

/// Compute pairwise squared Euclidean distance matrix between two sets of points.
pub fn distance_matrix(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = b.len();
    let mut dist = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            dist[i][j] = a[i].iter().zip(&b[j]).map(|(x, y)| (x - y).powi(2)).sum();
        }
    }
    dist
}

/// Compute the 1D Wasserstein-1 distance between two weighted histograms
/// on the same grid, using the cumulative distribution function.
pub fn wasserstein_1d_weighted(a_weights: &[f64], b_weights: &[f64]) -> f64 {
    assert_eq!(a_weights.len(), b_weights.len());
    let n = a_weights.len();
    let mut cdf_a = 0.0;
    let mut cdf_b = 0.0;
    let mut dist = 0.0;
    for i in 0..n {
        let _prev_a = cdf_a;
        let _prev_b = cdf_b;
        cdf_a += a_weights[i];
        cdf_b += b_weights[i];
        // The integral of |F_a - F_b| at this point
        dist += (cdf_a - cdf_b).abs();
    }
    dist
}

/// Wasserstein barycenter of K discrete distributions on a common support.
///
/// Uses iterative Bregman projection (Sinkhorn barycenter algorithm).
/// Each distribution is given as a weight vector over the same grid.
///
/// # Arguments
/// * `distributions` - Slice of weight vectors, each summing to 1.0
/// * `cost` - Cost matrix C where C[i][j] is the cost of moving from grid point i to j
/// * `reg` - Entropic regularization parameter
/// * `max_iter` - Maximum number of iterations
/// * `tol` - Convergence tolerance
///
/// # Returns
/// The barycenter as a weight vector over the same grid.
pub fn barycenter_sinkhorn(
    distributions: &[Vec<f64>],
    cost: &[Vec<f64>],
    reg: f64,
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    let n = cost.len();
    let k = distributions.len();
    assert!(k > 0, "Need at least one distribution");
    assert!(n > 0, "Need at least one support point");

    // Gibbs kernel K = exp(-C/reg)
    let kernel: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| (-cost[i][j] / reg).exp().max(1e-300))
                .collect()
        })
        .collect();

    // Dual variables for each distribution
    let mut vs: Vec<Vec<f64>> = (0..k).map(|_| vec![0.0; n]).collect();

    // Uniform initial barycenter
    let mut bary = vec![1.0 / n as f64; n];

    for _ in 0..max_iter {
        let old_bary = bary.clone();

        // Update each distribution's dual variable
        for dist_idx in 0..k {
            // u = bary / (K * v)
            let kv: Vec<f64> = (0..n)
                .map(|i| (0..n).map(|j| kernel[i][j] * vs[dist_idx][j].exp()).sum())
                .collect();

            let u: Vec<f64> = (0..n)
                .map(|i| {
                    if kv[i] > 1e-300 { bary[i] / kv[i] } else { 1e-300 }
                })
                .collect();

            // v = dist / (K^T * u)
            let ktu: Vec<f64> = (0..n)
                .map(|j| (0..n).map(|i| kernel[i][j] * u[i]).sum())
                .collect();

            for j in 0..n {
                if ktu[j] > 1e-300 {
                    vs[dist_idx][j] = (distributions[dist_idx][j] / ktu[j]).ln().max(-100.0);
                }
            }
        }

        // Update barycenter: geometric mean of all u_k * K * v_k
        bary = vec![0.0; n];
        for dist_idx in 0..k {
            let kv: Vec<f64> = (0..n)
                .map(|i| (0..n).map(|j| kernel[i][j] * vs[dist_idx][j].exp()).sum())
                .collect();
            for i in 0..n {
                bary[i] += kv[i].ln();
            }
        }
        for i in 0..n {
            bary[i] = (bary[i] / k as f64).exp();
        }

        // Normalize
        let sum: f64 = bary.iter().sum();
        if sum > 1e-300 {
            for x in bary.iter_mut() {
                *x /= sum;
            }
        }

        // Check convergence
        let change: f64 = bary.iter().zip(old_bary.iter()).map(|(a, b)| (a - b).abs()).sum();
        if change < tol {
            break;
        }
    }

    bary
}

/// Simple fixed-point barycenter for 1D distributions using the
/// quantile averaging method.
///
/// For 1D distributions on the real line, the Wasserstein barycenter is
/// simply the average of the quantile functions.
///
/// # Arguments
/// * `distributions` - Slice of sorted 1D sample arrays (each same length)
pub fn barycenter_1d_quantile(distributions: &[Vec<f64>]) -> Vec<f64> {
    assert!(!distributions.is_empty());
    let n = distributions[0].len();
    let k = distributions.len();

    let mut result = vec![0.0; n];
    for samples in distributions {
        assert_eq!(samples.len(), n, "All distributions must have equal size");
        for (i, &s) in samples.iter().enumerate() {
            result[i] += s;
        }
    }
    for x in result.iter_mut() {
        *x /= k as f64;
    }
    result
}

/// Compute the Wasserstein-2 cost from a candidate barycenter to all
/// input distributions (average squared distance).
pub fn barycenter_cost(barycenter: &[Vec<f64>], distributions: &[Vec<Vec<f64>>]) -> f64 {
    if distributions.is_empty() {
        return 0.0;
    }
    let mut total = 0.0;
    for dist in distributions {
        let d = dist_w2(barycenter, dist);
        total += d;
    }
    total / distributions.len() as f64
}

/// Wasserstein-2 distance between two equal-size point clouds.
pub fn dist_w2(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    if n == 0 {
        return 0.0;
    }
    let dim = a[0].len();
    let sum_sq: f64 = a.iter().zip(b.iter())
        .map(|(pa, pb)| {
            (0..dim).map(|d| (pa[d] - pb[d]).powi(2)).sum::<f64>()
        })
        .sum();
    (sum_sq / n as f64).sqrt()
}

/// Free-support Wasserstein barycenter via iterative displacement
/// interpolation (fixed-point iteration).
///
/// Each distribution is a point cloud of the same size and dimension.
/// The barycenter is computed by alternating between optimal assignment
/// and averaging.
///
/// # Arguments
/// * `distributions` - K point clouds, each with n points of dimension d
/// * `max_iter` - Maximum iterations
/// * `tol` - Convergence tolerance
pub fn barycenter_free_support(
    distributions: &[Vec<Vec<f64>>],
    max_iter: usize,
    tol: f64,
) -> Vec<Vec<f64>> {
    assert!(!distributions.is_empty());
    let k = distributions.len();
    let n = distributions[0].len();
    let dim = distributions[0][0].len();

    // Initialize barycenter as the first distribution
    let mut bary = distributions[0].clone();

    for _ in 0..max_iter {
        let old_bary = bary.clone();
        let mut new_bary = vec![vec![0.0; dim]; n];

        for dist in distributions {
            // Greedy assignment: sort both by first coordinate for approximate matching
            let mut indices: Vec<usize> = (0..n).collect();
            let mut bary_indices: Vec<usize> = (0..n).collect();

            indices.sort_by(|&i, &j| dist[i][0].partial_cmp(&dist[j][0]).unwrap_or(std::cmp::Ordering::Equal));
            bary_indices.sort_by(|&i, &j| old_bary[i][0].partial_cmp(&old_bary[j][0]).unwrap_or(std::cmp::Ordering::Equal));

            for (rank, &bi) in bary_indices.iter().enumerate() {
                let di = indices[rank];
                for d in 0..dim {
                    new_bary[bi][d] += dist[di][d];
                }
            }
        }

        for pt in new_bary.iter_mut() {
            for d in 0..dim {
                pt[d] /= k as f64;
            }
        }

        // Check convergence
        let change: f64 = new_bary.iter().zip(old_bary.iter())
            .map(|(a, b)| (0..dim).map(|d| (a[d] - b[d]).powi(2)).sum::<f64>())
            .sum::<f64>()
            .sqrt();

        bary = new_bary;
        if change < tol {
            break;
        }
    }

    bary
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_matrix_identity() {
        let pts = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let d = distance_matrix(&pts, &pts);
        for i in 0..3 {
            assert!((d[i][i]).abs() < 1e-10, "Diagonal should be zero");
        }
    }

    #[test]
    fn test_distance_matrix_symmetric() {
        let a = vec![vec![0.0], vec![2.0]];
        let b = vec![vec![1.0], vec![3.0]];
        let d = distance_matrix(&a, &b);
        assert!((d[0][0] - 1.0).abs() < 1e-10);
        assert!((d[0][1] - 9.0).abs() < 1e-10);
        assert!((d[1][0] - 1.0).abs() < 1e-10);
        assert!((d[1][1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_wasserstein_1d_weighted_identical() {
        let w = vec![0.25, 0.25, 0.25, 0.25];
        assert!((wasserstein_1d_weighted(&w, &w) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_wasserstein_1d_weighted_shifted() {
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 0.0, 1.0];
        let d = wasserstein_1d_weighted(&a, &b);
        assert!(d > 0.0, "Shifted distributions should have positive W1 distance");
    }

    #[test]
    fn test_barycenter_1d_quantile_midpoint() {
        let a = vec![0.0, 1.0, 2.0, 3.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let bary = barycenter_1d_quantile(&[a, b]);
        let expected = vec![1.0, 2.0, 3.0, 4.0];
        for i in 0..4 {
            assert!((bary[i] - expected[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_barycenter_1d_quantile_single() {
        let a = vec![1.0, 2.0, 3.0];
        let bary = barycenter_1d_quantile(&[a.clone()]);
        assert_eq!(bary, a);
    }

    #[test]
    fn test_barycenter_sinkhorn_uniform() {
        let n = 5;
        let uniform = vec![1.0 / n as f64; n];
        let dists = vec![uniform.clone(), uniform.clone()];
        let cost = vec![vec![0.0; n]; n]; // Zero cost (same point)
        let bary = barycenter_sinkhorn(&dists, &cost, 0.1, 100, 1e-8);
        let sum: f64 = bary.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Barycenter should sum to 1.0, got {}", sum);
    }

    #[test]
    fn test_barycenter_sinkhorn_positive() {
        let n = 4;
        let d1 = vec![0.5, 0.3, 0.1, 0.1];
        let d2 = vec![0.1, 0.1, 0.3, 0.5];
        let cost: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| (i as f64 - j as f64).powi(2)).collect()).collect();
        let bary = barycenter_sinkhorn(&[d1, d2], &cost, 0.01, 200, 1e-8);
        let sum: f64 = bary.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "Barycenter should sum to 1.0, got {}", sum);
        for &w in &bary {
            assert!(w >= 0.0, "Barycenter weights should be non-negative");
        }
    }

    #[test]
    fn test_dist_w2_identical() {
        let pts = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        assert!((dist_w2(&pts, &pts) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_dist_w2_shifted() {
        let a = vec![vec![0.0], vec![0.0]];
        let b = vec![vec![3.0], vec![3.0]];
        assert!((dist_w2(&a, &b) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_barycenter_cost_zero_for_identical() {
        let pts = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let dists = vec![pts.clone(), pts.clone()];
        assert!((barycenter_cost(&pts, &dists) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_barycenter_free_support_midpoint() {
        let a = vec![vec![0.0], vec![2.0]];
        let b = vec![vec![2.0], vec![4.0]];
        let bary = barycenter_free_support(&[a, b], 100, 1e-10);
        // After sorting, midpoint should be (1, 3)
        let mut sorted_bary = bary.clone();
        sorted_bary.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
        assert!((sorted_bary[0][0] - 1.0).abs() < 1e-6);
        assert!((sorted_bary[1][0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_barycenter_free_support_converges() {
        let a = vec![vec![0.0], vec![1.0]];
        let b = vec![vec![1.0], vec![2.0]];
        let bary1 = barycenter_free_support(&[a.clone(), b.clone()], 1, 1e-10);
        let bary10 = barycenter_free_support(&[a, b], 100, 1e-10);
        // With more iterations it should converge (both should be close to midpoint)
        let mut s1 = bary1.clone();
        let mut s10 = bary10.clone();
        s1.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
        s10.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
        assert!((s10[0][0] - 0.5).abs() < 0.5);
        assert!((s10[1][0] - 1.5).abs() < 0.5);
    }
}
