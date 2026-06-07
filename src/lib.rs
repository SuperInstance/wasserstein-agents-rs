#![allow(
    clippy::needless_range_loop,
    clippy::new_without_default,
    clippy::type_complexity,
    dead_code
)]
//! # Wasserstein Agents
//!
//! Optimal transport, Wasserstein distances, and agent distribution coordination.
//!
//! # Key Concepts
//! - **Sinkhorn algorithm**: Entropy-regularized optimal transport
//! - **Wasserstein distance**: Earth mover's distance between distributions
//! - **Agent distributions**: Multi-agent fleet coordination via transport plans
//! - **JKO scheme**: Wasserstein gradient flow for distribution evolution

mod agents;
mod gradient_flow;
mod transport;

pub use agents::AgentDistribution;
pub use gradient_flow::JKOScheme;
pub use transport::{OptimalTransport, SinkhornSolver};
