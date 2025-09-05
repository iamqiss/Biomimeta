//! Foveal Sampling Module

use ndarray::Array2;
use crate::AfiyahError;

/// Foveal sampling processor implementing biological foveal sampling
pub struct FovealSamplingProcessor {
    foveal_radius: f64,
    sampling_density: f64,
    peripheral_falloff: f64,
}

impl FovealSamplingProcessor {
    /// Creates a new foveal sampling processor
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            foveal_radius: 2.0,
            sampling_density: 1.0,
            peripheral_falloff: 0.8,
        })
    }

    /// Applies foveal sampling to input
    pub fn apply_sampling(&self, input: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        let (height, width) = input.dim();
        let mut sampled = Array2::zeros((height, width));

        for i in 0..height {
            for j in 0..width {
                let x = (j as f64) - (width as f64) / 2.0;
                let y = (i as f64) - (height as f64) / 2.0;
                let distance = (x * x + y * y).sqrt();

                let sampling_factor = if distance <= self.foveal_radius {
                    self.sampling_density
                } else {
                    self.sampling_density * (-distance / self.foveal_radius * self.peripheral_falloff).exp()
                };

                sampled[[i, j]] = input[[i, j]] * sampling_factor;
            }
        }

        Ok(sampled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_foveal_sampling_processor_creation() {
        let processor = FovealSamplingProcessor::new();
        assert!(processor.is_ok());
    }

    #[test]
    fn test_foveal_sampling() {
        let processor = FovealSamplingProcessor::new().unwrap();
        let input = Array2::ones((32, 32));
        
        let result = processor.apply_sampling(&input);
        assert!(result.is_ok());
        
        let sampled = result.unwrap();
        assert_eq!(sampled.dim(), (32, 32));
    }
}