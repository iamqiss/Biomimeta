//! Optic Flow Processing Module

use ndarray::Array2;
use crate::AfiyahError;

/// Flow vector representing motion in the visual field
#[derive(Debug, Clone, Copy)]
pub struct FlowVector {
    pub x: f64,
    pub y: f64,
    pub magnitude: f64,
    pub direction: f64,
    pub confidence: f64,
}

impl FlowVector {
    pub fn new(x: f64, y: f64, confidence: f64) -> Self {
        let magnitude = (x * x + y * y).sqrt();
        let direction = y.atan2(x);
        Self {
            x,
            y,
            magnitude,
            direction,
            confidence,
        }
    }

    pub fn zero() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            magnitude: 0.0,
            direction: 0.0,
            confidence: 0.0,
        }
    }
}

/// Flow field containing vectors for each spatial location
#[derive(Debug, Clone)]
pub struct FlowField {
    pub vectors: Array2<FlowVector>,
    pub global_flow: FlowVector,
    pub coherence: f64,
}

impl FlowField {
    pub fn new(height: usize, width: usize) -> Self {
        Self {
            vectors: Array2::from_shape_fn((height, width), |_| FlowVector::zero()),
            global_flow: FlowVector::zero(),
            coherence: 0.0,
        }
    }
}

impl Default for FlowField {
    fn default() -> Self {
        Self::new(64, 64)
    }
}

/// Optic flow processor implementing biological motion processing
pub struct OpticFlowProcessor {
    temporal_window: usize,
    spatial_window: usize,
    flow_threshold: f64,
}

impl OpticFlowProcessor {
    /// Creates a new optic flow processor
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            temporal_window: 3,
            spatial_window: 7,
            flow_threshold: 0.1,
        })
    }

    /// Computes optic flow between two frames
    pub fn compute_flow(&self, current: &Array2<f64>, previous: &Array2<f64>) -> Result<FlowField, AfiyahError> {
        let (height, width) = current.dim();
        let mut flow_field = FlowField::new(height, width);

        // Compute flow at each location
        for i in self.spatial_window/2..height-self.spatial_window/2 {
            for j in self.spatial_window/2..width-self.spatial_window/2 {
                let flow_vector = self.calculate_flow_vector(current, previous, i, j)?;
                flow_field.vectors[[i, j]] = flow_vector;
            }
        }

        // Calculate global flow and coherence
        self.calculate_global_flow(&mut flow_field)?;
        self.calculate_coherence(&mut flow_field)?;

        Ok(flow_field)
    }

    fn calculate_flow_vector(&self, current: &Array2<f64>, previous: &Array2<f64>, i: usize, j: usize) -> Result<FlowVector, AfiyahError> {
        let (height, width) = current.dim();
        let window_size = self.spatial_window;
        let mut best_dx = 0.0;
        let mut best_dy = 0.0;
        let mut best_correlation = 0.0;

        // Search for best match in local neighborhood
        for di in -2..=2 {
            for dj in -2..=2 {
                let ni = (i as i32 + di) as usize;
                let nj = (j as i32 + dj) as usize;

                if ni >= window_size/2 && ni < height - window_size/2 &&
                   nj >= window_size/2 && nj < width - window_size/2 {
                    
                    let correlation = self.calculate_correlation(current, previous, i, j, ni, nj)?;
                    
                    if correlation > best_correlation {
                        best_correlation = correlation;
                        best_dx = dj as f64;
                        best_dy = di as f64;
                    }
                }
            }
        }

        let confidence = best_correlation.min(1.0);
        Ok(FlowVector::new(best_dx, best_dy, confidence))
    }

    fn calculate_correlation(&self, current: &Array2<f64>, previous: &Array2<f64>, i: usize, j: usize, ni: usize, nj: usize) -> Result<f64, AfiyahError> {
        let window_size = self.spatial_window;
        let mut correlation = 0.0;
        let mut norm_current = 0.0;
        let mut norm_previous = 0.0;

        for di in -window_size/2..=window_size/2 {
            for dj in -window_size/2..=window_size/2 {
                let ci = i + di;
                let cj = j + dj;
                let pi = ni + di;
                let pj = nj + dj;

                if ci < current.nrows() && cj < current.ncols() &&
                   pi < previous.nrows() && pj < previous.ncols() {
                    
                    let curr_val = current[[ci, cj]];
                    let prev_val = previous[[pi, pj]];
                    
                    correlation += curr_val * prev_val;
                    norm_current += curr_val * curr_val;
                    norm_previous += prev_val * prev_val;
                }
            }
        }

        if norm_current > 0.0 && norm_previous > 0.0 {
            Ok(correlation / (norm_current * norm_previous).sqrt())
        } else {
            Ok(0.0)
        }
    }

    fn calculate_global_flow(&self, flow_field: &mut FlowField) -> Result<(), AfiyahError> {
        let mut weighted_x = 0.0;
        let mut weighted_y = 0.0;
        let mut total_weight = 0.0;

        for vector in flow_field.vectors.iter() {
            let weight = vector.confidence * vector.magnitude;
            weighted_x += vector.x * weight;
            weighted_y += vector.y * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            flow_field.global_flow = FlowVector::new(
                weighted_x / total_weight,
                weighted_y / total_weight,
                flow_field.coherence,
            );
        }

        Ok(())
    }

    fn calculate_coherence(&self, flow_field: &mut FlowField) -> Result<(), AfiyahError> {
        let mut total_magnitude = 0.0;
        let mut aligned_magnitude = 0.0;

        for vector in flow_field.vectors.iter() {
            total_magnitude += vector.magnitude;
            if vector.confidence > 0.5 {
                aligned_magnitude += vector.magnitude;
            }
        }

        if total_magnitude > 0.0 {
            flow_field.coherence = aligned_magnitude / total_magnitude;
        } else {
            flow_field.coherence = 0.0;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flow_vector_creation() {
        let vector = FlowVector::new(1.0, 1.0, 0.8);
        assert!((vector.magnitude - 2.0_f64.sqrt()).abs() < 1e-10);
        assert_eq!(vector.confidence, 0.8);
    }

    #[test]
    fn test_flow_field_creation() {
        let field = FlowField::new(32, 32);
        assert_eq!(field.vectors.dim(), (32, 32));
        assert_eq!(field.coherence, 0.0);
    }

    #[test]
    fn test_optic_flow_processor_creation() {
        let processor = OpticFlowProcessor::new();
        assert!(processor.is_ok());
    }

    #[test]
    fn test_flow_computation() {
        let processor = OpticFlowProcessor::new().unwrap();
        
        let current = Array2::ones((32, 32));
        let previous = Array2::zeros((32, 32));
        
        let result = processor.compute_flow(&current, &previous);
        assert!(result.is_ok());
        
        let flow_field = result.unwrap();
        assert!(flow_field.coherence >= 0.0);
    }
}