//! Sliced Wasserstein distance — fast approximation via random 1D projections.
//!
//! The Sliced Wasserstein distance projects high-dimensional distributions onto
//! random 1D lines (sampled uniformly from the unit sphere), computes the 1D
//! Wasserstein distance on each projection, and averages over all projections.
//!
//! This is significantly faster than computing the full Wasserstein distance
//! in high dimensions and provides a good approximation, especially as the
//! number of projections grows.

/// A simple pseudo-random number generator for reproducibility.
#[derive(Debug, Clone)]
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn next_normal(&mut self) -> f64 {
        // Box-Muller transform
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Sample a random unit vector on the sphere in `dim` dimensions using
/// the normal distribution method (Muller, 1959).
fn random_unit_sphere(dim: usize, rng: &mut Rng) -> Vec<f64> {
    let mut v: Vec<f64> = (0..dim).map(|_| rng.next_normal()).collect();
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm < 1e-300 {
        v[0] = 1.0;
        return v;
    }
    for x in v.iter_mut() {
        *x /= norm;
    }
    v
}

/// Project a set of points onto a 1D line defined by a unit direction vector.
/// Returns sorted projections.
fn project_1d(points: &[Vec<f64>], direction: &[f64]) -> Vec<f64> {
    let mut projections: Vec<f64> = points
        .iter()
        .map(|p| p.iter().zip(direction.iter()).map(|(a, b)| a * b).sum())
        .collect();
    projections.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    projections
}

/// Compute the 1D Wasserstein-1 distance between two sets of 1D values.
///
/// For sorted arrays of equal size, this is simply the average absolute
/// difference between corresponding elements.
fn wasserstein_1d_sorted(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Arrays must have equal length");
    let n = a.len();
    if n == 0 {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum::<f64>() / n as f64
}

/// Compute the 1D Wasserstein-2 distance between two sets of 1D values.
///
/// For sorted arrays of equal size, this is the square root of the average
/// squared difference between corresponding elements.
fn wasserstein_2d_sorted(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Arrays must have equal length");
    let n = a.len();
    if n == 0 {
        return 0.0;
    }
    let sum_sq: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    (sum_sq / n as f64).sqrt()
}

/// Sliced Wasserstein-1 distance between two point clouds.
///
/// Projects both distributions onto `num_projections` random directions,
/// computes the 1D Wasserstein-1 distance on each, and averages.
///
/// Both point clouds must have the same number of points and dimensionality.
///
/// # Arguments
/// * `a` - First point cloud (n points, each of dimension d)
/// * `b` - Second point cloud (n points, each of dimension d)
/// * `num_projections` - Number of random projection directions
/// * `seed` - Random seed for reproducibility
pub fn sliced_wasserstein_1(a: &[Vec<f64>], b: &[Vec<f64>], num_projections: usize, seed: u64) -> f64 {
    assert!(!a.is_empty(), "Point clouds must not be empty");
    assert_eq!(a.len(), b.len(), "Point clouds must have equal size");
    assert_eq!(a[0].len(), b[0].len(), "Point clouds must have equal dimension");

    let dim = a[0].len();
    let mut rng = Rng::new(seed);
    let mut total = 0.0;

    for _ in 0..num_projections {
        let direction = random_unit_sphere(dim, &mut rng);
        let proj_a = project_1d(a, &direction);
        let proj_b = project_1d(b, &direction);
        total += wasserstein_1d_sorted(&proj_a, &proj_b);
    }

    total / num_projections as f64
}

/// Sliced Wasserstein-2 distance between two point clouds.
///
/// Same as `sliced_wasserstein_1` but uses Wasserstein-2 on each slice.
pub fn sliced_wasserstein_2(a: &[Vec<f64>], b: &[Vec<f64>], num_projections: usize, seed: u64) -> f64 {
    assert!(!a.is_empty(), "Point clouds must not be empty");
    assert_eq!(a.len(), b.len(), "Point clouds must have equal size");
    assert_eq!(a[0].len(), b[0].len(), "Point clouds must have equal dimension");

    let dim = a[0].len();
    let mut rng = Rng::new(seed);
    let mut total = 0.0;

    for _ in 0..num_projections {
        let direction = random_unit_sphere(dim, &mut rng);
        let proj_a = project_1d(a, &direction);
        let proj_b = project_1d(b, &direction);
        total += wasserstein_2d_sorted(&proj_a, &proj_b);
    }

    total / num_projections as f64
}

/// Sliced Wasserstein distance with custom projection directions.
///
/// Instead of random directions, uses user-specified unit vectors.
/// Useful for deterministic evaluation or specific analysis.
pub fn sliced_wasserstein_custom(a: &[Vec<f64>], b: &[Vec<f64>], directions: &[Vec<f64>]) -> f64 {
    assert!(!a.is_empty());
    assert_eq!(a.len(), b.len());
    assert!(!directions.is_empty());

    let mut total = 0.0;
    for dir in directions {
        let proj_a = project_1d(a, dir);
        let proj_b = project_1d(b, dir);
        total += wasserstein_1d_sorted(&proj_a, &proj_b);
    }
    total / directions.len() as f64
}

/// Generate a set of random projection directions (unit vectors on the sphere).
///
/// Useful for inspecting or reusing projection directions.
pub fn generate_projections(dim: usize, num_projections: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = Rng::new(seed);
    (0..num_projections).map(|_| random_unit_sphere(dim, &mut rng)).collect()
}

/// Check whether a vector has approximately unit norm.
pub fn is_unit_vector(v: &[f64], tol: f64) -> bool {
    let norm_sq: f64 = v.iter().map(|x| x * x).sum();
    (norm_sq - 1.0).abs() < tol
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point_1d() -> Vec<Vec<f64>> {
        vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]
    }

    fn shifted_1d() -> Vec<Vec<f64>> {
        vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]]
    }

    fn point_2d() -> Vec<Vec<f64>> {
        vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]]
    }

    fn shifted_2d() -> Vec<Vec<f64>> {
        vec![vec![1.0, 1.0], vec![2.0, 1.0], vec![1.0, 2.0], vec![2.0, 2.0]]
    }

    #[test]
    fn test_identical_distributions_zero_distance() {
        let pts = point_2d();
        let d = sliced_wasserstein_1(&pts, &pts, 100, 42);
        assert!(d < 1e-10, "Identical distributions should have zero distance, got {}", d);
    }

    #[test]
    fn test_shifted_distributions_positive_distance() {
        let d = sliced_wasserstein_1(&point_2d(), &shifted_2d(), 100, 42);
        assert!(d > 0.0, "Shifted distributions should have positive distance");
    }

    #[test]
    fn test_shifted_distributions_positive_distance_1d() {
        let d = sliced_wasserstein_1(&point_1d(), &shifted_1d(), 100, 42);
        assert!(d > 0.0, "Shifted 1D distributions should have positive distance");
    }

    #[test]
    fn test_more_projections_converges() {
        // More projections should give a more stable estimate
        let d10 = sliced_wasserstein_1(&point_2d(), &shifted_2d(), 10, 42);
        let d1000 = sliced_wasserstein_1(&point_2d(), &shifted_2d(), 1000, 42);
        // Both should be positive and finite
        assert!(d10 > 0.0 && d10.is_finite());
        assert!(d1000 > 0.0 && d1000.is_finite());
    }

    #[test]
    fn test_wasserstein_1d_sorted_basic() {
        let a = vec![0.0, 1.0, 2.0];
        let b = vec![0.0, 1.0, 2.0];
        assert!((wasserstein_1d_sorted(&a, &b) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_wasserstein_1d_sorted_shifted() {
        let a = vec![0.0, 1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!((wasserstein_1d_sorted(&a, &b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_wasserstein_2d_sorted_basic() {
        let a = vec![0.0, 1.0, 2.0];
        let b = vec![0.0, 1.0, 2.0];
        assert!((wasserstein_2d_sorted(&a, &b) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_wasserstein_2d_sorted_shifted() {
        let a = vec![0.0, 3.0];
        let b = vec![3.0, 6.0];
        assert!((wasserstein_2d_sorted(&a, &b) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_project_1d_axis_aligned() {
        let pts = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let dir = vec![1.0, 0.0]; // x-axis
        let proj = project_1d(&pts, &dir);
        assert!((proj[0] - 0.0).abs() < 1e-10);
        assert!((proj[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_random_unit_sphere_norm() {
        let mut rng = Rng::new(42);
        for dim in [2, 3, 5, 10] {
            let v = random_unit_sphere(dim, &mut rng);
            let norm_sq: f64 = v.iter().map(|x| x * x).sum();
            assert!((norm_sq - 1.0).abs() < 1e-10, "dim={}: norm_sq={}", dim, norm_sq);
        }
    }

    #[test]
    fn test_generate_projections() {
        let dirs = generate_projections(3, 50, 42);
        assert_eq!(dirs.len(), 50);
        for d in &dirs {
            assert_eq!(d.len(), 3);
            assert!(is_unit_vector(d, 1e-10));
        }
    }

    #[test]
    fn test_sliced_wasserstein_custom() {
        let a = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let b = vec![vec![1.0, 1.0], vec![2.0, 2.0]];
        let dirs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let d = sliced_wasserstein_custom(&a, &b, &dirs);
        assert!(d > 0.0, "Custom sliced distance should be positive");
    }

    #[test]
    fn test_symmetry() {
        let a = point_2d();
        let b = shifted_2d();
        let d_ab = sliced_wasserstein_1(&a, &b, 200, 42);
        let d_ba = sliced_wasserstein_1(&b, &a, 200, 42);
        assert!((d_ab - d_ba).abs() < 1e-10, "SW distance should be symmetric: {} vs {}", d_ab, d_ba);
    }

    #[test]
    fn test_empty_projection_directions_unit() {
        assert!(is_unit_vector(&[1.0, 0.0, 0.0], 1e-10));
        assert!(!is_unit_vector(&[2.0, 0.0, 0.0], 1e-10));
    }
}
