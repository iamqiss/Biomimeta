//! Hebbian Learning Module

use ndarray::Array2;
use crate::AfiyahError;

/// Synaptic weight representing connection strength
#[derive(Debug, Clone, Copy)]
pub struct SynapticWeight {
    pub value: f64,
    pub max_value: f64,
    pub min_value: f64,
    pub learning_rate: f64,
}

impl SynapticWeight {
    pub fn new(value: f64, max_value: f64, min_value: f64, learning_rate: f64) -> Self {
        Self {
            value: value.clamp(min_value, max_value),
            max_value,
            min_value,
            learning_rate,
        }
    }

    pub fn update(&mut self, delta: f64) {
        self.value = (self.value + delta * self.learning_rate).clamp(self.min_value, self.max_value);
    }
}

/// Learning rule for Hebbian learning
#[derive(Debug, Clone, PartialEq)]
pub enum LearningRule {
    Hebbian,
    AntiHebbian,
    Covariance,
    BCM, // Bienenstock-Cooper-Munro
}

/// Hebbian learner implementing biological learning mechanisms
pub struct HebbianLearner {
    learning_rule: LearningRule,
    learning_rate: f64,
    weight_decay: f64,
    plasticity_threshold: f64,
}

impl HebbianLearner {
    /// Creates a new Hebbian learner
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            learning_rule: LearningRule::Hebbian,
            learning_rate: 0.01,
            weight_decay: 0.001,
            plasticity_threshold: 0.5,
        })
    }

    /// Learns from input using Hebbian learning
    pub fn learn(&mut self, input: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        let (height, width) = input.dim();
        let mut weights = Array2::zeros((height, width));

        // Initialize weights with small random values
        for i in 0..height {
            for j in 0..width {
                weights[[i, j]] = (rand::random::<f64>() - 0.5) * 0.1;
            }
        }

        // Apply Hebbian learning
        for i in 0..height {
            for j in 0..width {
                let input_value = input[[i, j]];
                let weight_delta = self.calculate_weight_delta(input_value, &weights, i, j)?;
                weights[[i, j]] += weight_delta;
            }
        }

        // Apply weight decay
        self.apply_weight_decay(&mut weights)?;

        Ok(weights)
    }

    fn calculate_weight_delta(&self, input_value: f64, weights: &Array2<f64>, i: usize, j: usize) -> Result<f64, AfiyahError> {
        match self.learning_rule {
            LearningRule::Hebbian => {
                // Classic Hebbian learning: Δw = η * x * y
                // For simplicity, we use input_value as both pre and post-synaptic activity
                Ok(self.learning_rate * input_value * input_value)
            },
            LearningRule::AntiHebbian => {
                // Anti-Hebbian learning: Δw = -η * x * y
                Ok(-self.learning_rate * input_value * input_value)
            },
            LearningRule::Covariance => {
                // Covariance learning: Δw = η * (x - <x>) * (y - <y>)
                let mean_input = self.calculate_mean_input(weights)?;
                let centered_input = input_value - mean_input;
                Ok(self.learning_rate * centered_input * centered_input)
            },
            LearningRule::BCM => {
                // BCM learning: Δw = η * x * y * (y - θ)
                let threshold = self.calculate_bcm_threshold(weights)?;
                let post_activity = input_value;
                if post_activity > threshold {
                    Ok(self.learning_rate * input_value * post_activity * (post_activity - threshold))
                } else {
                    Ok(0.0)
                }
            },
        }
    }

    fn calculate_mean_input(&self, weights: &Array2<f64>) -> Result<f64, AfiyahError> {
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

    fn calculate_bcm_threshold(&self, weights: &Array2<f64>) -> Result<f64, AfiyahError> {
        // BCM threshold is typically the square of the mean activity
        let mean_activity = self.calculate_mean_input(weights)?;
        Ok(mean_activity * mean_activity)
    }

    fn apply_weight_decay(&self, weights: &mut Array2<f64>) -> Result<(), AfiyahError> {
        for weight in weights.iter_mut() {
            *weight *= (1.0 - self.weight_decay);
        }
        Ok(())
    }

    /// Updates the learning rule
    pub fn set_learning_rule(&mut self, rule: LearningRule) {
        self.learning_rule = rule;
    }

    /// Updates the learning rate
    pub fn set_learning_rate(&mut self, rate: f64) {
        self.learning_rate = rate.clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synaptic_weight_creation() {
        let weight = SynapticWeight::new(0.5, 1.0, -1.0, 0.01);
        assert_eq!(weight.value, 0.5);
        assert_eq!(weight.max_value, 1.0);
        assert_eq!(weight.min_value, -1.0);
    }

    #[test]
    fn test_synaptic_weight_update() {
        let mut weight = SynapticWeight::new(0.5, 1.0, -1.0, 0.01);
        weight.update(0.1);
        assert!(weight.value > 0.5);
    }

    #[test]
    fn test_hebbian_learner_creation() {
        let learner = HebbianLearner::new();
        assert!(learner.is_ok());
    }

    #[test]
    fn test_hebbian_learning() {
        let mut learner = HebbianLearner::new().unwrap();
        let input = Array2::ones((32, 32));
        
        let result = learner.learn(&input);
        assert!(result.is_ok());
        
        let weights = result.unwrap();
        assert_eq!(weights.dim(), (32, 32));
    }

    #[test]
    fn test_learning_rule_update() {
        let mut learner = HebbianLearner::new().unwrap();
        learner.set_learning_rule(LearningRule::AntiHebbian);
        assert_eq!(learner.learning_rule, LearningRule::AntiHebbian);
    }
}