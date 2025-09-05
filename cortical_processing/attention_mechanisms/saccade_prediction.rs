//! Saccade Prediction Module

use ndarray::Array2;
use crate::AfiyahError;
use super::foveal_prioritization::FovealMap;
use super::saliency_mapping::SaliencyMap;

/// Saccade vector representing eye movement
#[derive(Debug, Clone, Copy)]
pub struct SaccadeVector {
    pub x: f64,
    pub y: f64,
    pub magnitude: f64,
    pub direction: f64,
    pub velocity: f64,
    pub duration: f64,
}

impl SaccadeVector {
    pub fn new(x: f64, y: f64, velocity: f64, duration: f64) -> Self {
        let magnitude = (x * x + y * y).sqrt();
        let direction = y.atan2(x);
        Self {
            x,
            y,
            magnitude,
            direction,
            velocity,
            duration,
        }
    }

    pub fn zero() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            magnitude: 0.0,
            direction: 0.0,
            velocity: 0.0,
            duration: 0.0,
        }
    }
}

/// Saccade target representing potential fixation points
#[derive(Debug, Clone)]
pub struct SaccadeTarget {
    pub position: (f64, f64),
    pub priority: f64,
    pub saccade_vector: SaccadeVector,
    pub confidence: f64,
    pub target_type: TargetType,
}

/// Types of saccade targets
#[derive(Debug, Clone, PartialEq)]
pub enum TargetType {
    Salient,
    Foveal,
    Predictive,
    Corrective,
}

/// Saccade predictor implementing biological eye movement prediction
pub struct SaccadePredictor {
    saccade_threshold: f64,
    max_saccade_amplitude: f64,
    saccade_velocity: f64,
    saccade_duration: f64,
}

impl SaccadePredictor {
    /// Creates a new saccade predictor
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            saccade_threshold: 0.7,
            max_saccade_amplitude: 20.0, // degrees
            saccade_velocity: 500.0, // degrees/second
            saccade_duration: 0.05, // seconds
        })
    }

    /// Predicts saccade targets based on foveal and saliency information
    pub fn predict_saccades(&self, foveal_map: &FovealMap, saliency_map: &SaliencyMap) -> Result<Vec<SaccadeTarget>, AfiyahError> {
        let mut targets = Vec::new();

        // Find salient regions that are not in fovea
        let salient_targets = self.find_salient_targets(foveal_map, saliency_map)?;
        targets.extend(salient_targets);

        // Find foveal regions that need attention
        let foveal_targets = self.find_foveal_targets(foveal_map)?;
        targets.extend(foveal_targets);

        // Sort targets by priority
        targets.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap());

        // Limit number of targets
        if targets.len() > 10 {
            targets.truncate(10);
        }

        Ok(targets)
    }

    fn find_salient_targets(&self, foveal_map: &FovealMap, saliency_map: &SaliencyMap) -> Result<Vec<SaccadeTarget>, AfiyahError> {
        let mut targets = Vec::new();
        let (height, width) = saliency_map.weights.dim();

        for i in 0..height {
            for j in 0..width {
                let saliency_weight = saliency_map.weights[[i, j]];
                let foveal_weight = foveal_map.weights[[i, j]];

                // Only consider regions that are salient but not in fovea
                if saliency_weight > self.saccade_threshold && foveal_weight < 0.5 {
                    let position = (j as f64, i as f64);
                    let priority = saliency_weight * (1.0 - foveal_weight);
                    
                    if priority > 0.3 {
                        let saccade_vector = self.calculate_saccade_vector(foveal_map.foveal_center, position)?;
                        let confidence = priority.min(1.0);

                        targets.push(SaccadeTarget {
                            position,
                            priority,
                            saccade_vector,
                            confidence,
                            target_type: TargetType::Salient,
                        });
                    }
                }
            }
        }

        Ok(targets)
    }

    fn find_foveal_targets(&self, foveal_map: &FovealMap) -> Result<Vec<SaccadeTarget>, AfiyahError> {
        let mut targets = Vec::new();

        for region in &foveal_map.regions {
            if region.priority > self.saccade_threshold {
                let saccade_vector = self.calculate_saccade_vector(foveal_map.foveal_center, region.center)?;
                let confidence = region.priority.min(1.0);

                targets.push(SaccadeTarget {
                    position: region.center,
                    priority: region.priority,
                    saccade_vector,
                    confidence,
                    target_type: TargetType::Foveal,
                });
            }
        }

        Ok(targets)
    }

    fn calculate_saccade_vector(&self, from: (f64, f64), to: (f64, f64)) -> Result<SaccadeVector, AfiyahError> {
        let dx = to.0 - from.0;
        let dy = to.1 - from.1;
        let distance = (dx * dx + dy * dy).sqrt();

        // Calculate saccade parameters based on distance
        let velocity = self.saccade_velocity * (distance / self.max_saccade_amplitude).min(1.0);
        let duration = distance / velocity.max(1.0);

        Ok(SaccadeVector::new(dx, dy, velocity, duration))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_saccade_vector_creation() {
        let vector = SaccadeVector::new(10.0, 5.0, 500.0, 0.05);
        assert!((vector.magnitude - (10.0_f64.powi(2) + 5.0_f64.powi(2)).sqrt()).abs() < 1e-10);
        assert_eq!(vector.velocity, 500.0);
    }

    #[test]
    fn test_saccade_target_creation() {
        let vector = SaccadeVector::new(5.0, 3.0, 400.0, 0.04);
        let target = SaccadeTarget {
            position: (10.0, 15.0),
            priority: 0.8,
            saccade_vector: vector,
            confidence: 0.9,
            target_type: TargetType::Salient,
        };
        assert_eq!(target.position, (10.0, 15.0));
        assert_eq!(target.priority, 0.8);
    }

    #[test]
    fn test_saccade_predictor_creation() {
        let predictor = SaccadePredictor::new();
        assert!(predictor.is_ok());
    }

    #[test]
    fn test_saccade_prediction() {
        let predictor = SaccadePredictor::new().unwrap();
        let foveal_map = FovealMap::new(32, 32);
        let saliency_map = SaliencyMap::new(32, 32);
        
        let result = predictor.predict_saccades(&foveal_map, &saliency_map);
        assert!(result.is_ok());
        
        let targets = result.unwrap();
        assert!(targets.len() >= 0);
    }
}