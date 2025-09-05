//! Foveal Prioritization Module

use ndarray::Array2;
use crate::AfiyahError;

/// Foveal region representing high-resolution processing area
#[derive(Debug, Clone)]
pub struct FovealRegion {
    pub center: (f64, f64),
    pub radius: f64,
    pub resolution_factor: f64,
    pub priority: f64,
}

impl FovealRegion {
    pub fn new(center: (f64, f64), radius: f64, resolution_factor: f64, priority: f64) -> Self {
        Self {
            center,
            radius,
            resolution_factor,
            priority,
        }
    }
}

/// Foveal map containing prioritization weights
#[derive(Debug, Clone)]
pub struct FovealMap {
    pub weights: Array2<f64>,
    pub regions: Vec<FovealRegion>,
    pub foveal_center: (f64, f64),
    pub foveal_radius: f64,
}

impl FovealMap {
    pub fn new(height: usize, width: usize) -> Self {
        Self {
            weights: Array2::zeros((height, width)),
            regions: Vec::new(),
            foveal_center: (width as f64 / 2.0, height as f64 / 2.0),
            foveal_radius: 2.0,
        }
    }

    pub fn calculate_strength(&self) -> Result<f64, AfiyahError> {
        let mut total_weight = 0.0;
        let mut count = 0;

        for weight in self.weights.iter() {
            total_weight += weight;
            count += 1;
        }

        if count > 0 {
            Ok(total_weight / count as f64)
        } else {
            Ok(0.0)
        }
    }
}

/// Foveal processor implementing biological foveal prioritization
pub struct FovealProcessor {
    foveal_radius: f64,
    resolution_falloff: f64,
    priority_threshold: f64,
}

impl FovealProcessor {
    /// Creates a new foveal processor
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            foveal_radius: 2.0, // degrees
            resolution_falloff: 0.8,
            priority_threshold: 0.3,
        })
    }

    /// Processes foveal prioritization for visual input
    pub fn process_foveal_prioritization(&self, input: &Array2<f64>) -> Result<FovealMap, AfiyahError> {
        let (height, width) = input.dim();
        let mut foveal_map = FovealMap::new(height, width);

        // Set foveal center (assume center of image for now)
        foveal_map.foveal_center = (width as f64 / 2.0, height as f64 / 2.0);
        foveal_map.foveal_radius = self.foveal_radius;

        // Calculate foveal weights based on distance from center
        for i in 0..height {
            for j in 0..width {
                let x = j as f64;
                let y = i as f64;
                let center_x = foveal_map.foveal_center.0;
                let center_y = foveal_map.foveal_center.1;

                let distance = ((x - center_x).powi(2) + (y - center_y).powi(2)).sqrt();
                let eccentricity = distance / (height as f64).min(width as f64) * 2.0; // Normalize to degrees

                // Calculate foveal weight based on eccentricity
                let foveal_weight = self.calculate_foveal_weight(eccentricity);
                foveal_map.weights[[i, j]] = foveal_weight;

                // Create foveal region if weight is high enough
                if foveal_weight > self.priority_threshold {
                    let resolution_factor = 1.0 / (1.0 + eccentricity * self.resolution_falloff);
                    let region = FovealRegion::new(
                        (x, y),
                        self.foveal_radius,
                        resolution_factor,
                        foveal_weight,
                    );
                    foveal_map.regions.push(region);
                }
            }
        }

        Ok(foveal_map)
    }

    /// Updates foveal center based on attention or eye movement
    pub fn update_foveal_center(&mut self, new_center: (f64, f64)) -> Result<(), AfiyahError> {
        // This would update the foveal center in a real implementation
        Ok(())
    }

    fn calculate_foveal_weight(&self, eccentricity: f64) -> f64 {
        // Calculate foveal weight based on eccentricity
        // Higher weight for lower eccentricity (closer to fovea)
        if eccentricity <= self.foveal_radius {
            1.0
        } else {
            // Exponential decay with eccentricity
            (-eccentricity / self.foveal_radius).exp()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_foveal_region_creation() {
        let region = FovealRegion::new((10.0, 20.0), 2.0, 1.0, 0.8);
        assert_eq!(region.center, (10.0, 20.0));
        assert_eq!(region.radius, 2.0);
        assert_eq!(region.priority, 0.8);
    }

    #[test]
    fn test_foveal_map_creation() {
        let foveal_map = FovealMap::new(32, 32);
        assert_eq!(foveal_map.weights.dim(), (32, 32));
        assert_eq!(foveal_map.regions.len(), 0);
    }

    #[test]
    fn test_foveal_processor_creation() {
        let processor = FovealProcessor::new();
        assert!(processor.is_ok());
    }

    #[test]
    fn test_foveal_processing() {
        let processor = FovealProcessor::new().unwrap();
        let input = Array2::ones((32, 32));
        
        let result = processor.process_foveal_prioritization(&input);
        assert!(result.is_ok());
        
        let foveal_map = result.unwrap();
        assert!(foveal_map.weights.iter().any(|&w| w > 0.0));
    }
}