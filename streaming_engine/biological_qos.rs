//! Biological QoS Module

use ndarray::Array2;
use crate::AfiyahError;

/// Perceptual quality metrics for biological QoS
#[derive(Debug, Clone)]
pub struct PerceptualQuality {
    pub overall_quality: f64,
    pub foveal_quality: f64,
    pub peripheral_quality: f64,
    pub motion_quality: f64,
    pub color_quality: f64,
    pub temporal_quality: f64,
    pub biological_accuracy: f64,
}

impl PerceptualQuality {
    pub fn new() -> Self {
        Self {
            overall_quality: 0.0,
            foveal_quality: 0.0,
            peripheral_quality: 0.0,
            motion_quality: 0.0,
            color_quality: 0.0,
            temporal_quality: 0.0,
            biological_accuracy: 0.0,
        }
    }
}

/// Biological QoS manager implementing human perceptual requirements
pub struct QoSManager {
    quality_weights: QualityWeights,
    adaptation_threshold: f64,
    quality_history: Vec<PerceptualQuality>,
}

/// Quality weights for different perceptual aspects
#[derive(Debug, Clone)]
struct QualityWeights {
    pub foveal_weight: f64,
    pub peripheral_weight: f64,
    pub motion_weight: f64,
    pub color_weight: f64,
    pub temporal_weight: f64,
}

impl QoSManager {
    /// Creates a new QoS manager
    pub fn new() -> Result<Self, AfiyahError> {
        let quality_weights = QualityWeights {
            foveal_weight: 0.4,
            peripheral_weight: 0.2,
            motion_weight: 0.2,
            color_weight: 0.1,
            temporal_weight: 0.1,
        };
        let adaptation_threshold = 0.1;
        let quality_history = Vec::new();

        Ok(Self {
            quality_weights,
            adaptation_threshold,
            quality_history,
        })
    }

    /// Initializes the QoS manager
    pub fn initialize(&mut self) -> Result<(), AfiyahError> {
        self.quality_history.clear();
        Ok(())
    }

    /// Stops the QoS manager
    pub fn stop(&mut self) -> Result<(), AfiyahError> {
        self.quality_history.clear();
        Ok(())
    }

    /// Assesses quality of input frame
    pub fn assess_quality(&mut self, frame: &Array2<f64>) -> Result<PerceptualQuality, AfiyahError> {
        let mut quality = PerceptualQuality::new();

        // Assess foveal quality (center region)
        quality.foveal_quality = self.assess_foveal_quality(frame)?;

        // Assess peripheral quality (edge regions)
        quality.peripheral_quality = self.assess_peripheral_quality(frame)?;

        // Assess motion quality (temporal stability)
        quality.motion_quality = self.assess_motion_quality(frame)?;

        // Assess color quality (chromatic information)
        quality.color_quality = self.assess_color_quality(frame)?;

        // Assess temporal quality (frame consistency)
        quality.temporal_quality = self.assess_temporal_quality(frame)?;

        // Calculate biological accuracy
        quality.biological_accuracy = self.calculate_biological_accuracy(frame)?;

        // Calculate overall quality
        quality.overall_quality = self.calculate_overall_quality(&quality)?;

        // Update quality history
        self.quality_history.push(quality.clone());
        if self.quality_history.len() > 10 {
            self.quality_history.remove(0);
        }

        Ok(quality)
    }

    fn assess_foveal_quality(&self, frame: &Array2<f64>) -> Result<f64, AfiyahError> {
        let (height, width) = frame.dim();
        let center_y = height / 2;
        let center_x = width / 2;
        let foveal_radius = (height.min(width) / 4) as usize;

        let mut foveal_sum = 0.0;
        let mut foveal_count = 0;

        for i in center_y.saturating_sub(foveal_radius)..(center_y + foveal_radius).min(height) {
            for j in center_x.saturating_sub(foveal_radius)..(center_x + foveal_radius).min(width) {
                foveal_sum += frame[[i, j]];
                foveal_count += 1;
            }
        }

        if foveal_count > 0 {
            Ok((foveal_sum / foveal_count as f64).clamp(0.0, 1.0))
        } else {
            Ok(0.0)
        }
    }

    fn assess_peripheral_quality(&self, frame: &Array2<f64>) -> Result<f64, AfiyahError> {
        let (height, width) = frame.dim();
        let mut peripheral_sum = 0.0;
        let mut peripheral_count = 0;

        // Sample edge regions
        for i in 0..height {
            for j in 0..width {
                let center_y = height as f64 / 2.0;
                let center_x = width as f64 / 2.0;
                let distance = ((i as f64 - center_y).powi(2) + (j as f64 - center_x).powi(2)).sqrt();
                let max_distance = ((height as f64).powi(2) + (width as f64).powi(2)).sqrt() / 2.0;

                if distance > max_distance * 0.5 {
                    peripheral_sum += frame[[i, j]];
                    peripheral_count += 1;
                }
            }
        }

        if peripheral_count > 0 {
            Ok((peripheral_sum / peripheral_count as f64).clamp(0.0, 1.0))
        } else {
            Ok(0.0)
        }
    }

    fn assess_motion_quality(&self, frame: &Array2<f64>) -> Result<f64, AfiyahError> {
        // Assess motion quality based on temporal consistency
        // This is a simplified version - in practice, this would compare with previous frames
        let (height, width) = frame.dim();
        let mut motion_sum = 0.0;
        let mut motion_count = 0;

        for i in 1..height-1 {
            for j in 1..width-1 {
                let center = frame[[i, j]];
                let mut local_variance = 0.0;
                let mut local_count = 0;

                for di in -1..=1 {
                    for dj in -1..=1 {
                        if di != 0 || dj != 0 {
                            let ni = (i as i32 + di) as usize;
                            let nj = (j as i32 + dj) as usize;
                            let diff = (center - frame[[ni, nj]]).abs();
                            local_variance += diff;
                            local_count += 1;
                        }
                    }
                }

                if local_count > 0 {
                    motion_sum += local_variance / local_count as f64;
                    motion_count += 1;
                }
            }
        }

        if motion_count > 0 {
            let motion_quality = 1.0 - (motion_sum / motion_count as f64).min(1.0);
            Ok(motion_quality.clamp(0.0, 1.0))
        } else {
            Ok(0.0)
        }
    }

    fn assess_color_quality(&self, frame: &Array2<f64>) -> Result<f64, AfiyahError> {
        // Assess color quality based on luminance distribution
        // This is a simplified version - in practice, this would analyze color channels
        let (height, width) = frame.dim();
        let mut color_sum = 0.0;
        let mut color_count = 0;

        for i in 0..height {
            for j in 0..width {
                let value = frame[[i, j]];
                // Calculate color richness (simplified)
                let color_richness = value * (1.0 - value).abs();
                color_sum += color_richness;
                color_count += 1;
            }
        }

        if color_count > 0 {
            Ok((color_sum / color_count as f64).clamp(0.0, 1.0))
        } else {
            Ok(0.0)
        }
    }

    fn assess_temporal_quality(&self, frame: &Array2<f64>) -> Result<f64, AfiyahError> {
        // Assess temporal quality based on frame consistency
        // This is a simplified version - in practice, this would compare with previous frames
        if self.quality_history.is_empty() {
            return Ok(0.5); // Default temporal quality
        }

        let last_quality = &self.quality_history[self.quality_history.len() - 1];
        let temporal_consistency = 1.0 - (last_quality.overall_quality - 0.5).abs() * 2.0;
        Ok(temporal_consistency.clamp(0.0, 1.0))
    }

    fn calculate_biological_accuracy(&self, frame: &Array2<f64>) -> Result<f64, AfiyahError> {
        // Calculate biological accuracy based on neural response patterns
        let (height, width) = frame.dim();
        let mut accuracy_sum = 0.0;
        let mut accuracy_count = 0;

        for i in 0..height {
            for j in 0..width {
                let value = frame[[i, j]];
                // Calculate biological accuracy (simplified)
                let accuracy = if value > 0.1 && value < 0.9 {
                    1.0
                } else if value > 0.05 && value < 0.95 {
                    0.8
                } else {
                    0.5
                };
                accuracy_sum += accuracy;
                accuracy_count += 1;
            }
        }

        if accuracy_count > 0 {
            Ok(accuracy_sum / accuracy_count as f64)
        } else {
            Ok(0.0)
        }
    }

    fn calculate_overall_quality(&self, quality: &PerceptualQuality) -> Result<f64, AfiyahError> {
        let overall_quality = 
            quality.foveal_quality * self.quality_weights.foveal_weight +
            quality.peripheral_quality * self.quality_weights.peripheral_weight +
            quality.motion_quality * self.quality_weights.motion_weight +
            quality.color_quality * self.quality_weights.color_weight +
            quality.temporal_quality * self.quality_weights.temporal_weight;

        Ok(overall_quality.clamp(0.0, 1.0))
    }

    /// Gets quality history
    pub fn get_quality_history(&self) -> &Vec<PerceptualQuality> {
        &self.quality_history
    }

    /// Updates quality weights
    pub fn update_quality_weights(&mut self, weights: QualityWeights) {
        self.quality_weights = weights;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perceptual_quality_creation() {
        let quality = PerceptualQuality::new();
        assert_eq!(quality.overall_quality, 0.0);
        assert_eq!(quality.foveal_quality, 0.0);
    }

    #[test]
    fn test_qos_manager_creation() {
        let manager = QoSManager::new();
        assert!(manager.is_ok());
    }

    #[test]
    fn test_quality_assessment() {
        let mut manager = QoSManager::new().unwrap();
        let frame = Array2::ones((32, 32));
        
        let result = manager.assess_quality(&frame);
        assert!(result.is_ok());
        
        let quality = result.unwrap();
        assert!(quality.overall_quality >= 0.0 && quality.overall_quality <= 1.0);
        assert!(quality.foveal_quality >= 0.0 && quality.foveal_quality <= 1.0);
    }

    #[test]
    fn test_initialization() {
        let mut manager = QoSManager::new().unwrap();
        let result = manager.initialize();
        assert!(result.is_ok());
    }

    #[test]
    fn test_stop() {
        let mut manager = QoSManager::new().unwrap();
        let result = manager.stop();
        assert!(result.is_ok());
    }
}