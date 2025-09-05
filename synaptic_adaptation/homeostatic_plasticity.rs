//! Homeostatic Plasticity Module

use ndarray::Array2;
use crate::AfiyahError;

/// Plasticity state representing the current state of synaptic plasticity
#[derive(Debug, Clone)]
pub struct PlasticityState {
    pub activity_level: f64,
    pub target_activity: f64,
    pub plasticity_factor: f64,
    pub adaptation_strength: f64,
}

impl PlasticityState {
    pub fn new() -> Self {
        Self {
            activity_level: 0.0,
            target_activity: 0.5,
            plasticity_factor: 1.0,
            adaptation_strength: 0.1,
        }
    }
}

/// Adaptation mechanism for homeostatic plasticity
#[derive(Debug, Clone, PartialEq)]
pub enum AdaptationMechanism {
    SynapticScaling,
    IntrinsicPlasticity,
    StructuralPlasticity,
    Metaplasticity,
}

/// Homeostatic plasticity processor implementing biological adaptation
pub struct HomeostaticPlasticity {
    adaptation_mechanism: AdaptationMechanism,
    target_activity: f64,
    adaptation_rate: f64,
    scaling_factor: f64,
    plasticity_threshold: f64,
}

impl HomeostaticPlasticity {
    /// Creates a new homeostatic plasticity processor
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            adaptation_mechanism: AdaptationMechanism::SynapticScaling,
            target_activity: 0.5,
            adaptation_rate: 0.01,
            scaling_factor: 1.0,
            plasticity_threshold: 0.3,
        })
    }

    /// Adapts weights using homeostatic plasticity
    pub fn adapt(&mut self, weights: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        let mut adapted_weights = weights.clone();
        let activity_level = self.calculate_activity_level(weights)?;
        
        // Calculate adaptation factor
        let adaptation_factor = self.calculate_adaptation_factor(activity_level)?;
        
        // Apply homeostatic adaptation
        match self.adaptation_mechanism {
            AdaptationMechanism::SynapticScaling => {
                self.apply_synaptic_scaling(&mut adapted_weights, adaptation_factor)?;
            },
            AdaptationMechanism::IntrinsicPlasticity => {
                self.apply_intrinsic_plasticity(&mut adapted_weights, adaptation_factor)?;
            },
            AdaptationMechanism::StructuralPlasticity => {
                self.apply_structural_plasticity(&mut adapted_weights, adaptation_factor)?;
            },
            AdaptationMechanism::Metaplasticity => {
                self.apply_metaplasticity(&mut adapted_weights, adaptation_factor)?;
            },
        }

        Ok(adapted_weights)
    }

    fn calculate_activity_level(&self, weights: &Array2<f64>) -> Result<f64, AfiyahError> {
        let mut total_activity = 0.0;
        let mut count = 0;

        for weight in weights.iter() {
            total_activity += weight.abs();
            count += 1;
        }

        if count > 0 {
            Ok(total_activity / count as f64)
        } else {
            Ok(0.0)
        }
    }

    fn calculate_adaptation_factor(&self, activity_level: f64) -> Result<f64, AfiyahError> {
        // Calculate adaptation factor based on deviation from target activity
        let deviation = activity_level - self.target_activity;
        let adaptation_factor = 1.0 + self.adaptation_rate * deviation;
        Ok(adaptation_factor.clamp(0.1, 2.0))
    }

    fn apply_synaptic_scaling(&self, weights: &mut Array2<f64>, scaling_factor: f64) -> Result<(), AfiyahError> {
        // Apply multiplicative scaling to all weights
        for weight in weights.iter_mut() {
            *weight *= scaling_factor;
        }
        Ok(())
    }

    fn apply_intrinsic_plasticity(&self, weights: &mut Array2<f64>, adaptation_factor: f64) -> Result<(), AfiyahError> {
        // Apply intrinsic plasticity by adjusting weight distribution
        let mean_weight = self.calculate_mean_weight(weights)?;
        let target_mean = mean_weight * adaptation_factor;
        
        for weight in weights.iter_mut() {
            *weight = (*weight - mean_weight) * adaptation_factor + target_mean;
        }
        Ok(())
    }

    fn apply_structural_plasticity(&self, weights: &mut Array2<f64>, adaptation_factor: f64) -> Result<(), AfiyahError> {
        // Apply structural plasticity by modifying weight structure
        let threshold = self.plasticity_threshold * adaptation_factor;
        
        for weight in weights.iter_mut() {
            if weight.abs() > threshold {
                *weight *= adaptation_factor;
            } else {
                *weight *= 0.5; // Reduce weak connections
            }
        }
        Ok(())
    }

    fn apply_metaplasticity(&self, weights: &mut Array2<f64>, adaptation_factor: f64) -> Result<(), AfiyahError> {
        // Apply metaplasticity by adjusting learning rates
        let learning_rate_factor = 1.0 / adaptation_factor;
        
        for weight in weights.iter_mut() {
            let sign = weight.signum();
            let magnitude = weight.abs();
            *weight = sign * magnitude.powf(learning_rate_factor);
        }
        Ok(())
    }

    fn calculate_mean_weight(&self, weights: &Array2<f64>) -> Result<f64, AfiyahError> {
        let mut sum = 0.0;
        let mut count = 0;

        for weight in weights.iter() {
            sum += weight;
            count += 1;
        }

        if count > 0 {
            Ok(sum / count as f64)
        } else {
            Ok(0.0)
        }
    }

    /// Updates the adaptation mechanism
    pub fn set_adaptation_mechanism(&mut self, mechanism: AdaptationMechanism) {
        self.adaptation_mechanism = mechanism;
    }

    /// Updates the target activity level
    pub fn set_target_activity(&mut self, target: f64) {
        self.target_activity = target.clamp(0.0, 1.0);
    }

    /// Updates the adaptation rate
    pub fn set_adaptation_rate(&mut self, rate: f64) {
        self.adaptation_rate = rate.clamp(0.0, 0.1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plasticity_state_creation() {
        let state = PlasticityState::new();
        assert_eq!(state.activity_level, 0.0);
        assert_eq!(state.target_activity, 0.5);
    }

    #[test]
    fn test_homeostatic_plasticity_creation() {
        let plasticity = HomeostaticPlasticity::new();
        assert!(plasticity.is_ok());
    }

    #[test]
    fn test_homeostatic_adaptation() {
        let mut plasticity = HomeostaticPlasticity::new().unwrap();
        let weights = Array2::ones((32, 32));
        
        let result = plasticity.adapt(&weights);
        assert!(result.is_ok());
        
        let adapted_weights = result.unwrap();
        assert_eq!(adapted_weights.dim(), (32, 32));
    }

    #[test]
    fn test_adaptation_mechanism_update() {
        let mut plasticity = HomeostaticPlasticity::new().unwrap();
        plasticity.set_adaptation_mechanism(AdaptationMechanism::IntrinsicPlasticity);
        assert_eq!(plasticity.adaptation_mechanism, AdaptationMechanism::IntrinsicPlasticity);
    }

    #[test]
    fn test_target_activity_update() {
        let mut plasticity = HomeostaticPlasticity::new().unwrap();
        plasticity.set_target_activity(0.7);
        assert_eq!(plasticity.target_activity, 0.7);
    }
}