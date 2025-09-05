//! Perceptual Error Model Module

use ndarray::Array2;
use crate::AfiyahError;

/// Perceptual error model implementing biological error perception
pub struct PerceptualErrorModel {
    error_threshold: f64,
    sensitivity_factor: f64,
    adaptation_rate: f64,
}

impl PerceptualErrorModel {
    /// Creates a new perceptual error model
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            error_threshold: 0.1,
            sensitivity_factor: 1.0,
            adaptation_rate: 0.05,
        })
    }

    /// Calculates perceptual error between reference and distorted images
    pub fn calculate_perceptual_error(&self, reference: &Array2<f64>, distorted: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        let (height, width) = reference.dim();
        let mut error = Array2::zeros((height, width));

        for i in 0..height {
            for j in 0..width {
                let diff = (reference[[i, j]] - distorted[[i, j]]).abs();
                let perceptual_error = if diff > self.error_threshold {
                    diff * self.sensitivity_factor
                } else {
                    0.0
                };
                error[[i, j]] = perceptual_error;
            }
        }

        Ok(error)
    }

    /// Updates the error threshold
    pub fn set_error_threshold(&mut self, threshold: f64) {
        self.error_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Updates the sensitivity factor
    pub fn set_sensitivity_factor(&mut self, factor: f64) {
        self.sensitivity_factor = factor.clamp(0.1, 10.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perceptual_error_model_creation() {
        let model = PerceptualErrorModel::new();
        assert!(model.is_ok());
    }

    #[test]
    fn test_perceptual_error_calculation() {
        let model = PerceptualErrorModel::new().unwrap();
        let reference = Array2::ones((32, 32));
        let distorted = Array2::ones((32, 32));
        
        let result = model.calculate_perceptual_error(&reference, &distorted);
        assert!(result.is_ok());
        
        let error = result.unwrap();
        assert_eq!(error.dim(), (32, 32));
    }

    #[test]
    fn test_error_threshold_update() {
        let mut model = PerceptualErrorModel::new().unwrap();
        model.set_error_threshold(0.2);
        assert_eq!(model.error_threshold, 0.2);
    }
}