//! Heading Direction Module

use crate::AfiyahError;
use super::motion_processing::MotionField;
use super::optic_flow::FlowField;

/// Heading vector representing direction of motion
#[derive(Debug, Clone, Copy)]
pub struct HeadingVector {
    pub x: f64,
    pub y: f64,
    pub magnitude: f64,
    pub direction: f64,
    pub confidence: f64,
}

impl HeadingVector {
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

/// Navigation field for spatial navigation
#[derive(Debug, Clone)]
pub struct NavigationField {
    pub heading_vectors: Vec<HeadingVector>,
    pub global_heading: HeadingVector,
    pub navigation_confidence: f64,
}

impl NavigationField {
    pub fn new() -> Self {
        Self {
            heading_vectors: Vec::new(),
            global_heading: HeadingVector::zero(),
            navigation_confidence: 0.0,
        }
    }
}

/// Heading detector implementing biological navigation processing
pub struct HeadingDetector {
    integration_window: usize,
    heading_threshold: f64,
}

impl HeadingDetector {
    /// Creates a new heading detector
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            integration_window: 5,
            heading_threshold: 0.3,
        })
    }

    /// Estimates heading direction from motion and flow information
    pub fn estimate_heading(&self, motion_field: &MotionField, flow_field: &FlowField) -> Result<HeadingVector, AfiyahError> {
        let mut heading_x = 0.0;
        let mut heading_y = 0.0;
        let mut total_weight = 0.0;

        // Integrate motion and flow information
        for (motion_vec, flow_vec) in motion_field.vectors.iter().zip(flow_field.vectors.iter()) {
            let motion_weight = motion_vec.confidence * motion_vec.magnitude;
            let flow_weight = flow_vec.confidence * flow_vec.magnitude;
            let combined_weight = motion_weight + flow_weight;

            if combined_weight > self.heading_threshold {
                heading_x += (motion_vec.x * motion_weight + flow_vec.x * flow_weight) / combined_weight;
                heading_y += (motion_vec.y * motion_weight + flow_vec.y * flow_weight) / combined_weight;
                total_weight += combined_weight;
            }
        }

        let confidence = if total_weight > 0.0 {
            (total_weight / (motion_field.vectors.len() as f64)).min(1.0)
        } else {
            0.0
        };

        if total_weight > 0.0 {
            Ok(HeadingVector::new(
                heading_x / total_weight,
                heading_y / total_weight,
                confidence,
            ))
        } else {
            Ok(HeadingVector::zero())
        }
    }

    /// Creates a navigation field for spatial navigation
    pub fn create_navigation_field(&self, motion_field: &MotionField, flow_field: &FlowField) -> Result<NavigationField, AfiyahError> {
        let mut nav_field = NavigationField::new();

        // Extract heading vectors from motion and flow
        for (motion_vec, flow_vec) in motion_field.vectors.iter().zip(flow_field.vectors.iter()) {
            let motion_weight = motion_vec.confidence * motion_vec.magnitude;
            let flow_weight = flow_vec.confidence * flow_vec.magnitude;
            let combined_weight = motion_weight + flow_weight;

            if combined_weight > self.heading_threshold {
                let heading_x = (motion_vec.x * motion_weight + flow_vec.x * flow_weight) / combined_weight;
                let heading_y = (motion_vec.y * motion_weight + flow_vec.y * flow_weight) / combined_weight;
                let confidence = combined_weight.min(1.0);

                nav_field.heading_vectors.push(HeadingVector::new(
                    heading_x,
                    heading_y,
                    confidence,
                ));
            }
        }

        // Calculate global heading
        nav_field.global_heading = self.estimate_heading(motion_field, flow_field)?;

        // Calculate navigation confidence
        nav_field.navigation_confidence = self.calculate_navigation_confidence(&nav_field)?;

        Ok(nav_field)
    }

    fn calculate_navigation_confidence(&self, nav_field: &NavigationField) -> Result<f64, AfiyahError> {
        if nav_field.heading_vectors.is_empty() {
            return Ok(0.0);
        }

        let mut total_confidence = 0.0;
        let mut count = 0;

        for heading in &nav_field.heading_vectors {
            total_confidence += heading.confidence;
            count += 1;
        }

        if count > 0 {
            Ok(total_confidence / count as f64)
        } else {
            Ok(0.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::motion_processing::MotionField;
    use super::super::optic_flow::FlowField;

    #[test]
    fn test_heading_vector_creation() {
        let vector = HeadingVector::new(1.0, 1.0, 0.8);
        assert!((vector.magnitude - 2.0_f64.sqrt()).abs() < 1e-10);
        assert_eq!(vector.confidence, 0.8);
    }

    #[test]
    fn test_navigation_field_creation() {
        let field = NavigationField::new();
        assert_eq!(field.heading_vectors.len(), 0);
        assert_eq!(field.navigation_confidence, 0.0);
    }

    #[test]
    fn test_heading_detector_creation() {
        let detector = HeadingDetector::new();
        assert!(detector.is_ok());
    }

    #[test]
    fn test_heading_estimation() {
        let detector = HeadingDetector::new().unwrap();
        let motion_field = MotionField::default();
        let flow_field = FlowField::default();
        
        let result = detector.estimate_heading(&motion_field, &flow_field);
        assert!(result.is_ok());
        
        let heading = result.unwrap();
        assert!(heading.confidence >= 0.0);
    }
}