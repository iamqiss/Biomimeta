//! Object Recognition Module

use ndarray::Array2;
use crate::AfiyahError;

/// Object features extracted from visual input
#[derive(Debug, Clone)]
pub struct ObjectFeatures {
    pub shapes: Vec<ShapeDescriptor>,
    pub overall_complexity: f64,
}

impl ObjectFeatures {
    pub fn new() -> Self {
        Self {
            shapes: Vec::new(),
            overall_complexity: 0.0,
        }
    }

    pub fn calculate_strength(&self) -> Result<f64, AfiyahError> {
        let mut total_strength = 0.0;
        let mut count = 0;

        for shape in &self.shapes {
            total_strength += shape.confidence;
            count += 1;
        }

        if count > 0 {
            Ok(total_strength / count as f64)
        } else {
            Ok(0.0)
        }
    }
}

/// Shape descriptor for object recognition
#[derive(Debug, Clone)]
pub struct ShapeDescriptor {
    pub shape_type: ShapeType,
    pub orientation: f64,
    pub scale: f64,
    pub position: (f64, f64),
    pub confidence: f64,
    pub complexity: f64,
}

/// Types of shapes that can be detected
#[derive(Debug, Clone, PartialEq)]
pub enum ShapeType {
    Circle,
    Rectangle,
    Triangle,
    Ellipse,
    Polygon,
    Line,
    Curve,
    Complex,
}

/// Object detector implementing biological object recognition
pub struct ObjectDetector {
    feature_threshold: f64,
}

impl ObjectDetector {
    /// Creates a new object detector with biological parameters
    pub fn new() -> Result<Self, AfiyahError> {
        Ok(Self {
            feature_threshold: 0.3,
        })
    }

    /// Extracts object features from visual input
    pub fn extract_features(&self, input: &Array2<f64>) -> Result<ObjectFeatures, AfiyahError> {
        let mut features = ObjectFeatures::new();

        // Extract shape features
        let shapes = self.detect_shapes(input)?;
        features.shapes = shapes;

        // Calculate overall complexity
        features.overall_complexity = self.calculate_complexity(&features)?;

        Ok(features)
    }

    fn detect_shapes(&self, input: &Array2<f64>) -> Result<Vec<ShapeDescriptor>, AfiyahError> {
        let mut shapes = Vec::new();
        let (height, width) = input.dim();

        // Simple shape detection based on local patterns
        for i in 1..height-1 {
            for j in 1..width-1 {
                let center = input[[i, j]];
                let mut pattern = 0u8;
                
                // Check 3x3 neighborhood
                for di in -1..=1 {
                    for dj in -1..=1 {
                        if di != 0 || dj != 0 {
                            let ni = (i as i32 + di) as usize;
                            let nj = (j as i32 + dj) as usize;
                            if input[[ni, nj]] > center {
                                pattern |= 1 << ((di + 1) * 3 + (dj + 1));
                            }
                        }
                    }
                }

                // Classify pattern
                let shape_type = self.classify_pattern(pattern);
                let confidence = self.calculate_confidence(pattern);

                if confidence > self.feature_threshold {
                    shapes.push(ShapeDescriptor {
                        shape_type,
                        orientation: 0.0,
                        scale: 1.0,
                        position: (j as f64, i as f64),
                        confidence,
                        complexity: confidence,
                    });
                }
            }
        }

        Ok(shapes)
    }

    fn classify_pattern(&self, pattern: u8) -> ShapeType {
        // Simple pattern classification
        match pattern {
            0b00000000 | 0b11111111 => ShapeType::Smooth,
            0b00010000 | 0b11101111 => ShapeType::Line,
            0b00010100 | 0b11101011 => ShapeType::Corner,
            0b00011100 | 0b11100011 => ShapeType::Edge,
            _ => ShapeType::Complex,
        }
    }

    fn calculate_confidence(&self, pattern: u8) -> f64 {
        // Calculate confidence based on pattern strength
        let ones = pattern.count_ones() as f64;
        let total = 8.0;
        let uniformity = 1.0 - (ones / total - 0.5).abs() * 2.0;
        uniformity
    }

    fn calculate_complexity(&self, features: &ObjectFeatures) -> Result<f64, AfiyahError> {
        let mut complexity = 0.0;
        complexity += features.shapes.len() as f64 * 0.3;
        Ok(complexity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_object_features_creation() {
        let features = ObjectFeatures::new();
        assert_eq!(features.shapes.len(), 0);
        assert_eq!(features.overall_complexity, 0.0);
    }

    #[test]
    fn test_object_detector_creation() {
        let detector = ObjectDetector::new();
        assert!(detector.is_ok());
    }

    #[test]
    fn test_feature_extraction() {
        let detector = ObjectDetector::new().unwrap();
        let input = Array2::ones((32, 32));
        
        let result = detector.extract_features(&input);
        assert!(result.is_ok());
        
        let features = result.unwrap();
        assert!(features.overall_complexity >= 0.0);
    }
}