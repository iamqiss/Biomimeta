//! Saliency Mapping Module

use ndarray::Array2;
use crate::AfiyahError;

/// Saliency region representing areas of high visual interest
#[derive(Debug, Clone)]
pub struct SaliencyRegion {
    pub center: (f64, f64),
    pub radius: f64,
    pub saliency: f64,
    pub feature_type: FeatureType,
}

/// Types of salient features
#[derive(Debug, Clone, PartialEq)]
pub enum FeatureType {
    Intensity,
    Color,
    Orientation,
    Motion,
    Texture,
    Edge,
}

/// Saliency map containing attention weights
#[derive(Debug, Clone)]
pub struct SaliencyMap {
    pub weights: Array2<f64>,
    pub regions: Vec<SaliencyRegion>,
    pub global_saliency: f64,
}

impl SaliencyMap {
    pub fn new(height: usize, width: usize) -> Self {
        Self {
            weights: Array2::zeros((height, width)),
            regions: Vec::new(),
            global_saliency: 0.0,
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

/// Saliency processor implementing biological saliency computation
pub struct SaliencyProcessor {
    intensity_weight: f64,
    color_weight: f64,
    orientation_weight: f64,
    motion_weight: f64,
    saliency_threshold: f64,
}

impl SaliencyProcessor {
    /// Creates a new saliency processor
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            intensity_weight: 0.3,
            color_weight: 0.2,
            orientation_weight: 0.2,
            motion_weight: 0.3,
            saliency_threshold: 0.5,
        })
    }

    /// Computes saliency map for visual input
    pub fn compute_saliency(&self, input: &Array2<f64>) -> Result<SaliencyMap, AfiyahError> {
        let (height, width) = input.dim();
        let mut saliency_map = SaliencyMap::new(height, width);

        // Compute intensity saliency
        let intensity_saliency = self.compute_intensity_saliency(input)?;

        // Compute color saliency (placeholder - would need color input)
        let color_saliency = self.compute_color_saliency(input)?;

        // Compute orientation saliency
        let orientation_saliency = self.compute_orientation_saliency(input)?;

        // Compute motion saliency (placeholder - would need temporal input)
        let motion_saliency = self.compute_motion_saliency(input)?;

        // Combine saliency maps
        for i in 0..height {
            for j in 0..width {
                let combined_saliency = 
                    intensity_saliency[[i, j]] * self.intensity_weight +
                    color_saliency[[i, j]] * self.color_weight +
                    orientation_saliency[[i, j]] * self.orientation_weight +
                    motion_saliency[[i, j]] * self.motion_weight;

                saliency_map.weights[[i, j]] = combined_saliency;

                // Create saliency region if saliency is high enough
                if combined_saliency > self.saliency_threshold {
                    let region = SaliencyRegion {
                        center: (j as f64, i as f64),
                        radius: 2.0,
                        saliency: combined_saliency,
                        feature_type: self.classify_feature_type(
                            intensity_saliency[[i, j]],
                            color_saliency[[i, j]],
                            orientation_saliency[[i, j]],
                            motion_saliency[[i, j]],
                        ),
                    };
                    saliency_map.regions.push(region);
                }
            }
        }

        // Calculate global saliency
        saliency_map.global_saliency = saliency_map.calculate_strength()?;

        Ok(saliency_map)
    }

    fn compute_intensity_saliency(&self, input: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        let (height, width) = input.dim();
        let mut saliency = Array2::zeros((height, width));

        // Compute local intensity contrast
        for i in 1..height-1 {
            for j in 1..width-1 {
                let center = input[[i, j]];
                let mut contrast = 0.0;
                let mut count = 0;

                // Check 3x3 neighborhood
                for di in -1..=1 {
                    for dj in -1..=1 {
                        if di != 0 || dj != 0 {
                            let ni = (i as i32 + di) as usize;
                            let nj = (j as i32 + dj) as usize;
                            contrast += (center - input[[ni, nj]]).abs();
                            count += 1;
                        }
                    }
                }

                if count > 0 {
                    saliency[[i, j]] = contrast / count as f64;
                }
            }
        }

        Ok(saliency)
    }

    fn compute_color_saliency(&self, input: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        // Placeholder for color saliency computation
        // In a real implementation, this would analyze color contrast
        let (height, width) = input.dim();
        Ok(Array2::zeros((height, width)))
    }

    fn compute_orientation_saliency(&self, input: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        let (height, width) = input.dim();
        let mut saliency = Array2::zeros((height, width));

        // Compute orientation contrast using simple edge detection
        for i in 1..height-1 {
            for j in 1..width-1 {
                let gx = input[[i, j+1]] - input[[i, j-1]];
                let gy = input[[i+1, j]] - input[[i-1, j]];
                let magnitude = (gx * gx + gy * gy).sqrt();
                saliency[[i, j]] = magnitude;
            }
        }

        Ok(saliency)
    }

    fn compute_motion_saliency(&self, input: &Array2<f64>) -> Result<Array2<f64>, AfiyahError> {
        // Placeholder for motion saliency computation
        // In a real implementation, this would analyze temporal changes
        let (height, width) = input.dim();
        Ok(Array2::zeros((height, width)))
    }

    fn classify_feature_type(&self, intensity: f64, color: f64, orientation: f64, motion: f64) -> FeatureType {
        let max_val = intensity.max(color).max(orientation).max(motion);
        
        if max_val == intensity {
            FeatureType::Intensity
        } else if max_val == color {
            FeatureType::Color
        } else if max_val == orientation {
            FeatureType::Orientation
        } else if max_val == motion {
            FeatureType::Motion
        } else {
            FeatureType::Intensity
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_saliency_region_creation() {
        let region = SaliencyRegion {
            center: (10.0, 20.0),
            radius: 2.0,
            saliency: 0.8,
            feature_type: FeatureType::Intensity,
        };
        assert_eq!(region.center, (10.0, 20.0));
        assert_eq!(region.saliency, 0.8);
    }

    #[test]
    fn test_saliency_map_creation() {
        let saliency_map = SaliencyMap::new(32, 32);
        assert_eq!(saliency_map.weights.dim(), (32, 32));
        assert_eq!(saliency_map.regions.len(), 0);
    }

    #[test]
    fn test_saliency_processor_creation() {
        let processor = SaliencyProcessor::new();
        assert!(processor.is_ok());
    }

    #[test]
    fn test_saliency_computation() {
        let processor = SaliencyProcessor::new().unwrap();
        let input = Array2::ones((32, 32));
        
        let result = processor.compute_saliency(&input);
        assert!(result.is_ok());
        
        let saliency_map = result.unwrap();
        assert!(saliency_map.weights.iter().any(|&w| w >= 0.0));
    }
}